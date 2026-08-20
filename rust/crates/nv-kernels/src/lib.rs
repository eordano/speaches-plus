#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub mod sys {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[cfg(feature = "cuda")]
pub mod graph;

#[cfg(feature = "cuda")]
pub mod lora;

pub mod lora_meta;

#[cfg(feature = "cuda")]
extern "C" {
    pub fn nv_kernels_lora_shrink(
        stream: *mut std::ffi::c_void,
        x_bf16: *const u16,
        a_ptrs: *const u64,
        buffer: *mut f32,
        token_lora_mapping: *const i32,
        token_indices_sorted: *const i32,
        num_tokens_per_lora: *const i32,
        lora_token_start_loc: *const i32,
        active_lora_ids: *const i32,
        m: i32,
        rank: i32,
        k: i32,
        n_slices: i32,
        grid_loras: i32,
        a_d0_stride: i64,
        scale: f32,
    ) -> i32;

    pub fn nv_kernels_lora_fused(
        stream: *mut std::ffi::c_void,
        x_bf16: *const u16,
        a_ptrs: *const u64,
        b_ptrs: *const u64,
        y_bf16: *mut u16,
        token_indices_sorted: *const i32,
        num_tokens_per_lora: *const i32,
        lora_token_start_loc: *const i32,
        active_lora_ids: *const i32,
        slice_n: *const i32,
        slice_start: *const i32,
        b_d0_stride: *const i64,
        m: i32,
        rank: i32,
        k: i32,
        max_n: i32,
        n_slices: i32,
        grid_loras: i32,
        a_d0_stride: i64,
        win_off: i32,
        win_len: i32,
        y_row_stride: i32,
        scale: f32,
    ) -> i32;

    pub fn nv_kernels_lora_expand(
        stream: *mut std::ffi::c_void,
        buffer: *const f32,
        b_ptrs: *const u64,
        y_bf16: *mut u16,
        token_lora_mapping: *const i32,
        token_indices_sorted: *const i32,
        num_tokens_per_lora: *const i32,
        lora_token_start_loc: *const i32,
        active_lora_ids: *const i32,
        slice_n: *const i32,
        slice_start: *const i32,
        m: i32,
        rank: i32,
        max_n: i32,
        n_slices: i32,
        grid_loras: i32,
        y_row_stride: i32,
    ) -> i32;
}

pub mod shift_decode_fold;

#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub mod cuda {
    use super::sys;
    use std::ffi::c_void;

    macro_rules! launchers_forward_to_the_c_symbol_verbatim {
        ($($name:ident = $sym:ident ( $($arg:ident : $ty:ty),* );)*) => {
            $(pub unsafe fn $name($($arg: $ty),*) -> i32 {
                sys::$sym($($arg),*)
            })*
        };
    }

    launchers_forward_to_the_c_symbol_verbatim! {
        hello_launch = nv_kernels_hello_launch(stream: *mut c_void, out: *mut f32, n: usize);
        rmsnorm_f32 = nv_kernels_rmsnorm_f32(stream: *mut c_void, x: *const f32, weight: *const f32, y: *mut f32, batch: usize, hidden: usize, eps: f32);
        rmsnorm_bf16 = nv_kernels_rmsnorm_bf16(stream: *mut c_void, x: *const u16, weight: *const u16, y: *mut u16, batch: usize, hidden: usize, eps: f32);
        rope_f32 = nv_kernels_rope_f32(stream: *mut c_void, q: *mut f32, k: *mut f32, cos_tbl: *const f32, sin_tbl: *const f32, positions: *const i32, batch: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize);
        rope_bf16 = nv_kernels_rope_bf16(stream: *mut c_void, q: *mut u16, k: *mut u16, cos_tbl: *const f32, sin_tbl: *const f32, positions: *const i32, batch: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize);
        silu_f32 = nv_kernels_silu_f32(stream: *mut c_void, x: *const f32, y: *mut f32, n: usize);
        silu_bf16 = nv_kernels_silu_bf16(stream: *mut c_void, x: *const u16, y: *mut u16, n: usize);
        silu_mul_f32 = nv_kernels_silu_mul_f32(stream: *mut c_void, x: *const f32, gate: *const f32, y: *mut f32, n: usize);
        silu_mul_bf16 = nv_kernels_silu_mul_bf16(stream: *mut c_void, x: *const u16, gate: *const u16, y: *mut u16, n: usize);
        silu_mul_bf16_candle = nv_kernels_silu_mul_bf16_candle(stream: *mut c_void, x: *const u16, gate: *const u16, y: *mut u16, n: usize);
        gelu_tanh_mul_bf16 = nv_kernels_gelu_tanh_mul_bf16(stream: *mut c_void, gate: *const u16, up: *const u16, y: *mut u16, n: usize);
        gemv_bf16 = nv_kernels_gemv_bf16(stream: *mut c_void, w: *const u16, x: *const u16, y: *mut u16, n: i32, k: i32);
        gemm_bf16_mk = nv_kernels_gemm_bf16_mk(stream: *mut c_void, w: *const u16, x: *const u16, y: *mut u16, n: i32, k: i32, m: i32);
        gemv_w4a16 = nv_kernels_gemv_w4a16(stream: *mut c_void, packed: *const u32, scale: *const u16, x: *const u16, y: *mut u16, n: i32, k: i32, gs: i32);
        attn_decode_f32 = nv_kernels_attn_decode_f32(stream: *mut c_void, q: *const f32, k: *const f32, v: *const f32, out: *mut f32, nh: i32, nkv: i32, hd: i32, total: i32, start: i32);
        incr_pos = nv_kernels_incr_pos(stream: *mut c_void, pos: *mut i32);
        write_kv_f32 = nv_kernels_write_kv_f32(stream: *mut c_void, src_k: *const f32, src_v: *const f32, cache_k: *mut f32, cache_v: *mut f32, pos: *const i32, nkv: i32, hd: i32);
        attn_decode_dev_f32 = nv_kernels_attn_decode_dev_f32(stream: *mut c_void, q: *const f32, k: *const f32, v: *const f32, out: *mut f32, pos: *const i32, nh: i32, nkv: i32, hd: i32, window: i32);
        flash_decode_dev_f32 = nv_kernels_flash_decode_dev_f32(stream: *mut c_void, q: *const f32, k: *const f32, v: *const f32, out: *mut f32, pos: *const i32, nh: i32, nkv: i32, hd: i32, window: i32);
        cast_bf16_f32 = nv_kernels_cast_bf16_f32(stream: *mut c_void, x: *const u16, y: *mut f32, n: i32);
        cast_f32_bf16 = nv_kernels_cast_f32_bf16(stream: *mut c_void, x: *const f32, y: *mut u16, n: i32);
        rms_no_weight_bf16_f32 = nv_kernels_rms_no_weight_bf16_f32(stream: *mut c_void, x: *const u16, y: *mut f32, rows: i32, dim: i32, eps: f32);
        gelu_mul_bf16f32 = nv_kernels_gelu_mul_bf16f32(stream: *mut c_void, gate: *const u16, pli: *const f32, y: *mut u16, n: i32);
        cast_scale_bf16_f32 = nv_kernels_cast_scale_bf16_f32(stream: *mut c_void, x: *const u16, y: *mut f32, scale: f32, n: i32);
        add_scale_f32 = nv_kernels_add_scale_f32(stream: *mut c_void, a: *const f32, b: *const f32, y: *mut f32, scale: f32, n: i32);
        incr_pos_rope = nv_kernels_incr_pos_rope(stream: *mut c_void, pos: *mut i32, rope_pos: *mut i32);
        argmax_bf16 = nv_kernels_argmax_bf16(stream: *mut c_void, logits: *const u16, n: i32, part_val: *mut f32, part_idx: *mut i32, pos: *const i32, token_out: *mut u32, ring: *mut u32, ring_mask: i32);
        rmsnorm_bf16w_f32out = nv_kernels_rmsnorm_bf16w_f32out(stream: *mut c_void, x: *const u16, w: *const u16, y: *mut f32, rows: i32, dim: i32, eps: f32);
        rmsnorm_add_scale_bf16 = nv_kernels_rmsnorm_add_scale_bf16(stream: *mut c_void, x: *const u16, w: *const u16, res: *const u16, y: *mut u16, rstd_out: *mut f32, next_w: *const u16, normed_out: *mut u16, rows: i32, dim: i32, eps: f32, scale: f32, eps_next: f32);
        qkv_prep = nv_kernels_qkv_prep(stream: *mut c_void, qkv: *const u16, qw: *const u16, kw: *const u16, cos_tbl: *const f32, sin_tbl: *const f32, rope_pos: *const i32, cache_pos: *const i32, delta: i32, q_out: *mut f32, kcache: *mut u16, vcache: *mut u16, nh: i32, nkv: i32, hd: i32, eps: f32);
        rstd_bf16 = nv_kernels_rstd_bf16(stream: *mut c_void, x: *const u16, rstd_out: *mut f32, rows: i32, dim: i32, eps: f32);
        rms_apply_bf16 = nv_kernels_rms_apply_bf16(stream: *mut c_void, x: *const u16, w: *const u16, rstd: *const f32, y: *mut u16, n: i32, dim: i32);
        gemv_w4a16_gelu_pli = nv_kernels_gemv_w4a16_gelu_pli(stream: *mut c_void, packed: *const u32, scale: *const u16, x: *const u16, pli: *const f32, y: *mut u16, n: i32, k: i32, gs: i32);
        rowquant_i8 = nv_kernels_rowquant_i8(stream: *mut c_void, w: *const u16, wq: *mut i8, row_scale: *mut f32, n: i32, k: i32);
        gemv_i8_normed = nv_kernels_gemv_i8_normed(stream: *mut c_void, wq: *const i8, row_scale: *const f32, x: *const u16, wn: *const u16, rstd: *const f32, y: *mut u16, n: i32, k: i32);
        gemv_bf16_normed = nv_kernels_gemv_bf16_normed(stream: *mut c_void, w: *const u16, x: *const u16, wn: *const u16, rstd: *const f32, y: *mut u16, n: i32, k: i32);
        flash_decode_dev_f32_bf16out = nv_kernels_flash_decode_dev_f32_bf16out(stream: *mut c_void, q: *const f32, k: *const f32, v: *const f32, out: *mut u16, pos: *const i32, nh: i32, nkv: i32, hd: i32, window: i32);
        flash_decode_splitk_bf16kv = nv_kernels_flash_decode_splitk_bf16kv(stream: *mut c_void, q: *const f32, k: *const u16, v: *const u16, out: *mut u16, pos: *const i32, scratch: *mut f32, nh: i32, nkv: i32, hd: i32, window: i32);
        flash_decode_fused_bf16kv = nv_kernels_flash_decode_fused_bf16kv(stream: *mut c_void, q: *const f32, k: *const u16, v: *const u16, out: *mut u16, pos: *const i32, delta: i32, scratch: *mut f32, fan_in: *mut u32, nh: i32, nkv: i32, hd: i32, window: i32);
        write_kv_bf16 = nv_kernels_write_kv_bf16(stream: *mut c_void, src_k: *const f32, src_v: *const f32, cache_k: *mut u16, cache_v: *mut u16, pos: *const i32, nkv: i32, hd: i32);
        marlin_workspace_elems = nv_kernels_marlin_workspace_elems(out_elems: *mut i32);
        marlin_gemm_w4a16_ex = nv_kernels_marlin_gemm_w4a16_ex(stream: *mut c_void, a: *const c_void, b_q_marlin: *const c_void, b_scales: *const c_void, c_out: *mut c_void, c_tmp: *mut c_void, workspace: *mut c_void, m: i32, n: i32, k: i32, group_size: i32, a_is_bf16: i32, use_atomic_add: i32, use_fp32_reduce: i32);
        marlin_repack_w4a16 = nv_kernels_marlin_repack_w4a16(stream: *mut c_void, b_q_packed: *const c_void, b_q_marlin_out: *mut c_void, k: i32, n: i32, num_bits: i32);
        marlin_gemm_w4a16 = nv_kernels_marlin_gemm_w4a16(stream: *mut c_void, a: *const c_void, b_q_marlin: *const c_void, b_scales: *const c_void, c_out: *mut c_void, workspace: *mut c_void, m: i32, n: i32, k: i32, group_size: i32, a_is_bf16: i32);
        marlin_gemm_w4a16_prezeroed = nv_kernels_marlin_gemm_w4a16_prezeroed(stream: *mut c_void, a: *const c_void, b_q_marlin: *const c_void, b_scales: *const c_void, c_out: *mut c_void, workspace: *mut c_void, m: i32, n: i32, k: i32, group_size: i32, a_is_bf16: i32);
        multi_zero_bf16 = nv_kernels_multi_zero_bf16(stream: *mut c_void, list: *const c_void, n: i32);
        kv_append_bf16 = nv_kernels_kv_append_bf16(stream: *mut c_void, k_new: *const u16, v_new: *const u16, kc: *mut u16, vc: *mut u16, n_committed: *const i32, k: i32, nkv: i32, hd: i32);
        tree_verify_attn_bf16 = nv_kernels_tree_verify_attn_bf16(stream: *mut c_void, q: *const u16, kc: *const u16, vc: *const u16, n_committed: *const i32, mask: *const u8, positions: *const i32, out: *mut u16, nh: i32, nkv: i32, hd: i32, k: i32, window: i32);
        gqa512_verify_bf16 = nv_kernels_gqa512_verify_bf16(stream: *mut c_void, q: *const u16, k: *const u16, v: *const u16, out: *mut u16, pos: *const i32, delta: i32, m: i32, scratch: *mut f32, nh: i32, nkv: i32, hd: i32, splits: i32);
        gqa512_verify_fp8 = nv_kernels_gqa512_verify_fp8(stream: *mut c_void, q: *const u16, k_fp8: *const u8, v_fp8: *const u8, k_scale: *const f32, v_scale: *const f32, out: *mut u16, pos: *const i32, delta: i32, m: i32, scratch: *mut f32, nh: i32, nkv: i32, hd: i32, splits: i32, scaling: f32);
        kv_compact_bf16 = nv_kernels_kv_compact_bf16(stream: *mut c_void, kc: *mut u16, vc: *mut u16, sk: *mut u16, sv: *mut u16, path: *const i32, base: i32, a: i32, stride: i32);
        tree_verify_attn_fp8 = nv_kernels_tree_verify_attn_fp8(stream: *mut c_void, q: *const u16, kc: *const u8, vc: *const u8, k_scale: *const f32, v_scale: *const f32, n_committed: *const i32, mask: *const u8, positions: *const i32, out: *mut u16, nh: i32, nkv: i32, hd: i32, k: i32, window: i32, ring: i32);
        kv_append_fp8 = nv_kernels_kv_append_fp8(stream: *mut c_void, k_new: *const u16, v_new: *const u16, kc: *mut u8, vc: *mut u8, k_scale: *mut f32, v_scale: *mut f32, n_committed: *const i32, k: i32, nkv: i32, hd: i32, ring: i32);
        verify_qkv_prep = nv_kernels_verify_qkv_prep(stream: *mut c_void, qkv: *const u16, qkv_stride: i64, q_off: i64, k_off: i64, v_off: i64, qw: *const u16, kw: *const u16, vw: *const u16, eps: f32, cos_tbl: *const f32, sin_tbl: *const f32, positions: *const i32, q_out: *mut u16, kc: *mut u8, vc: *mut u8, k_scale: *mut f32, v_scale: *mut f32, n_committed: *const i32, k: i32, nq: i32, nkv: i32, hd: i32, ring: i32);
        rmsnorm2_residual_bf16 = nv_kernels_rmsnorm2_residual_bf16(stream: *mut c_void, x: *const u16, residual: *const u16, w1: *const u16, w2: *const u16, sum_out: *mut u16, normed_out: *mut u16, batch: usize, hidden: usize, eps: f32);
        rmsnorm_residual_scale_bf16 = nv_kernels_rmsnorm_residual_scale_bf16(stream: *mut c_void, x: *const u16, residual: *const u16, w: *const u16, out: *mut u16, batch: usize, hidden: usize, eps: f32, scale: f32);
        kv_compact_fp8 = nv_kernels_kv_compact_fp8(stream: *mut c_void, kc: *mut u8, vc: *mut u8, k_scale: *mut f32, v_scale: *mut f32, sk: *mut u8, sv: *mut u8, ssk: *mut f32, ssv: *mut f32, path: *const i32, base: i32, a: i32, nkv: i32, hd: i32, ring: i32);
        gemv_i8_normed_mk = nv_kernels_gemv_i8_normed_mk(stream: *mut c_void, wq: *const i8, row_scale: *const f32, x: *const u16, wn: *const u16, rstd: *const f32, y: *mut u16, n: i32, k: i32, m: i32);
        gemv_i8_mk_h = nv_kernels_gemv_i8_mk_h(stream: *mut c_void, wq: *const i8, row_scale: *const f32, x: *const u16, wn: *const u16, rstd: *const f32, y: *mut u16, n: i32, k: i32, m: i32);
        normx_mk = nv_kernels_normx_mk(stream: *mut c_void, x: *const u16, wn: *const u16, rstd: *const f32, xn: *mut u16, k: i32, m: i32);
        gemv_i8_prenormed_mk = nv_kernels_gemv_i8_prenormed_mk(stream: *mut c_void, wq: *const i8, row_scale: *const f32, xn: *const u16, y: *mut u16, n: i32, k: i32, m: i32);
        rowquant_e4m3 = nv_kernels_rowquant_e4m3(stream: *mut c_void, w: *const u16, wq: *mut u8, row_scale: *mut f32, n: i32, k: i32);
        gemv_e4m3_mk_h = nv_kernels_gemv_e4m3_mk_h(stream: *mut c_void, wq: *const u8, row_scale: *const f32, x: *const u16, wn: *const u16, rstd: *const f32, y: *mut u16, n: i32, k: i32, m: i32);
        gemv_e4m3_mk = nv_kernels_gemv_e4m3_mk(stream: *mut c_void, wq: *const u8, row_scale: *const f32, x: *const u16, y: *mut u16, n: i32, k: i32, m: i32);
        scale_rowcol_bf16 = nv_kernels_scale_rowcol_bf16(stream: *mut c_void, d: *mut u16, row_scale_m: *const f32, col_scale_n: *const f32, m: i32, n: i32);
        residual_add_scale_bf16 = nv_kernels_residual_add_scale_bf16(stream: *mut c_void, a: *const u16, b: *const u16, y: *mut u16, scale: f32, n: usize);
        scale_inplace_bf16 = nv_kernels_scale_inplace_bf16(stream: *mut c_void, y: *mut u16, scale: f32, n: usize);
        scale_out_bf16 = nv_kernels_scale_out_bf16(stream: *mut c_void, x: *const u16, y: *mut u16, scale: f32, n: usize);
        gelu_tanh_mul_fused_bf16 = nv_kernels_gelu_tanh_mul_fused_bf16(stream: *mut c_void, fused: *const u16, y: *mut u16, inter: i32, tot_pairs: usize);
        tanh_softcap_bf16_to_f32 = nv_kernels_tanh_softcap_bf16_to_f32(stream: *mut c_void, x: *const u16, y: *mut f32, cap: f32, n: usize);
        nvfp4_quantize_row_bf16 = nv_kernels_nvfp4_quantize_row_bf16(stream: *mut c_void, x: *const u16, packed_out: *mut u8, scales_out: *mut u8, stored_global: f32, k: i32);
        nvfp4_gemv_bf16 = nv_kernels_nvfp4_gemv_bf16(stream: *mut c_void, w_packed: *const u8, w_scales: *const u8, x_packed: *const u8, x_scales: *const u8, y: *mut u16, alpha: f32, n: i32, k: i32);
        nvfp4_gemv_bf16act = nv_kernels_nvfp4_gemv_bf16act(stream: *mut c_void, w_packed: *const u8, w_scales: *const u8, x_bf16: *const u16, y: *mut u16, alpha: f32, n: i32, k: i32);
        gemv_nvfp4_w4a16_dual_m1 = nv_kernels_gemv_nvfp4_w4a16_dual_m1(stream: *mut c_void, wq_a: *const u8, sc_a: *const u8, wq_b: *const u8, sc_b: *const u8, x: *const u16, y_a: *mut u16, y_b: *mut u16, alpha_a: f32, alpha_b: f32, n: i32, k: i32);
        gemm_nvfp4_w4a16_mk_dual = nv_kernels_gemm_nvfp4_w4a16_mk_dual(stream: *mut c_void, wq_a: *const u8, sc_a: *const u8, wq_b: *const u8, sc_b: *const u8, x: *const u16, y_a: *mut u16, y_b: *mut u16, alpha_a: f32, alpha_b: f32, m: i32, n: i32, k: i32);
        gemv_nvfp4_w4a16_silu_gate_up_in_m1 = nv_kernels_gemv_nvfp4_w4a16_silu_gate_up_in_m1(stream: *mut c_void, wq: *const u8, sc: *const u8, gate: *const u16, up: *const u16, y: *mut u16, alpha: f32, n: i32, k: i32);
        gemv_nvfp4_w4a8_dual_m1 = nv_kernels_gemv_nvfp4_w4a8_dual_m1(stream: *mut c_void, wq_a: *const u8, sc_a: *const u8, wq_b: *const u8, sc_b: *const u8, x_q8: *const i8, x_dequant_scale: *const f32, y_a: *mut u16, y_b: *mut u16, alpha_a: f32, alpha_b: f32, n: i32, k: i32);
        silu_mul_rowquant_i8_m1 = nv_kernels_silu_mul_rowquant_i8_m1(stream: *mut c_void, gate: *const u16, up: *const u16, act_q8: *mut i8, act_dequant_scale: *mut f32, k: i32);
        silu_mul_rowquant_i8_mk = nv_kernels_silu_mul_rowquant_i8_mk(stream: *mut c_void, gate: *const u16, up: *const u16, act_staged_bf16: *mut u16, partial_absmax: *mut f32, act_q8: *mut i8, act_dequant_scales: *mut f32, m: i32, k: i32);
        silu_mul_stage_partial_absmax_m1 = nv_kernels_silu_mul_stage_partial_absmax_m1(stream: *mut c_void, gate: *const u16, up: *const u16, act_staged_bf16: *mut u16, partial_absmax: *mut f32, k: i32);
        gemv_nvfp4_w4a8_down_residual_quant_prologue_m1 = nv_kernels_gemv_nvfp4_w4a8_down_residual_quant_prologue_m1(stream: *mut c_void, wq: *const u8, sc: *const u8, act_staged_bf16: *const u16, partial_absmax: *const f32, num_partials: i32, residual: *const u16, y: *mut u16, alpha: f32, n: i32, k: i32);
        rmsnorm_residual_writeout_rowquant_i8_m1 = nv_kernels_rmsnorm_residual_writeout_rowquant_i8_m1(stream: *mut c_void, x: *const u16, res_in: *const u16, weight: *const u16, res_out: *mut u16, out_q8: *mut i8, out_dequant_scale: *mut f32, hidden: i32, eps: f32);
        gemv_e4m3_qkv_one_m1 = nv_kernels_gemv_e4m3_qkv_one_m1(stream: *mut c_void, wq_q: *const u8, rs_q: *const f32, wq_k: *const u8, rs_k: *const f32, wq_v: *const u8, rs_v: *const f32, x: *const u16, y_q: *mut u16, y_k: *mut u16, y_v: *mut u16, n_q: i32, n_k: i32, n_v: i32, k: i32);
        gemv_nvfp4_w4a8_dual_mk = nv_kernels_gemv_nvfp4_w4a8_dual_mk(stream: *mut c_void, wq_a: *const u8, sc_a: *const u8, wq_b: *const u8, sc_b: *const u8, x_q8: *const i8, x_dequant_scales: *const f32, y_a: *mut u16, y_b: *mut u16, alpha_a: f32, alpha_b: f32, m: i32, n: i32, k: i32);
        gemv_nvfp4_w4a8_down_residual_mk = nv_kernels_gemv_nvfp4_w4a8_down_residual_mk(stream: *mut c_void, wq: *const u8, sc: *const u8, x_q8: *const i8, x_dequant_scales: *const f32, residual: *const u16, y: *mut u16, alpha: f32, m: i32, n: i32, k: i32);
        gemv_nvfp4_w4a8_down_residual_m1 = nv_kernels_gemv_nvfp4_w4a8_down_residual_m1(stream: *mut c_void, wq: *const u8, sc: *const u8, x_q8: *const i8, x_dequant_scale: *const f32, residual: *const u16, y: *mut u16, alpha: f32, n: i32, k: i32);
        gemv_nvfp4_w4a8_down_residual_m1_rstd_emit = nv_kernels_gemv_nvfp4_w4a8_down_residual_m1_rstd_emit(stream: *mut c_void, wq: *const u8, sc: *const u8, x_q8: *const i8, x_dequant_scale: *const f32, residual: *const u16, y: *mut u16, alpha: f32, rstd_ssq_count_pack: *mut f32, rstd_eps: f32, n: i32, k: i32);
        qkv_norm_rope_kvstore_fp8_decode = nv_kernels_qkv_norm_rope_kvstore_fp8_decode(stream: *mut c_void, q_raw: *const u16, k_raw: *const u16, v_raw: *const u16, q_norm_w: *const u16, k_norm_w: *const u16, cos_tab: *const f32, sin_tab: *const f32, pos_dev: *const i32, k_fp8_base: *mut u8, v_fp8_base: *mut u8, k_scales_base: *mut f32, v_scales_base: *mut f32, q_out: *mut u16, q_sig_out: *mut u16, n_q: i32, n_kv: i32, hd: i32, q_row_stride: i32, rotary_dim: i32, eps: f32);
        quantize_kv_fp8 = nv_kernels_quantize_kv_fp8(stream: *mut c_void, x_bf16: *const u16, x_fp8_base: *mut u8, scales_base: *mut f32, start_dev: *const i32, n_tokens: i32, n_kv: i32, head_dim: i32, ring: i32);
        dequantize_kv_fp8 = nv_kernels_dequantize_kv_fp8(stream: *mut c_void, x_fp8: *const u8, scales: *const f32, x_bf16: *mut u16, start: i32, n_tokens: i32, n_kv: i32, head_dim: i32, ring: i32);
        quantize_kv_fp8_paged = nv_kernels_quantize_kv_fp8_paged(stream: *mut c_void, x_bf16: *const u16, x_fp8_base: *mut u8, scales_base: *mut f32, start_dev: *const i32, block_table: *const i32, block_size: i32, n_tokens: i32, n_kv: i32, head_dim: i32);
        dequantize_kv_fp8_paged = nv_kernels_dequantize_kv_fp8_paged(stream: *mut c_void, x_fp8_base: *const u8, scales_base: *const f32, x_bf16_out: *mut u16, block_table: *const i32, block_size: i32, len: i32, n_kv: i32, head_dim: i32);
        derive_v_from_k_fp8_paged = nv_kernels_derive_v_from_k_fp8_paged(stream: *mut c_void, k_fp8_base: *const u8, k_scales_base: *const f32, cos_tab: *const f32, sin_tab: *const f32, inv_freq: *const f32, v_bf16_out: *mut u16, block_table: *const i32, block_size: i32, len: i32, n_kv: i32, head_dim: i32, rope_angles: i32, angle_mode: i32, pos_base: i32, w_inv: f32);
        copy_kv_block_fp8 = nv_kernels_copy_kv_block_fp8(stream: *mut c_void, fp8_base: *const u8, scales_base: *const f32, fp8_dst_base: *mut u8, scales_dst_base: *mut f32, src_block: i32, dst_block: i32, block_size: i32, n_kv: i32, head_dim: i32);
        attention_fp8_decode = nv_kernels_attention_fp8_decode(stream: *mut c_void, q: *const u16, k_fp8: *const u8, v_fp8: *const u8, k_scales: *const f32, v_scales: *const f32, out: *mut u16, n_q: i32, n_kv: i32, head_dim: i32, n_total_dev: *const i32, max_total: i32, sliding_window: i32, scaling: f32);
        attention_fp8_decode_gscores = nv_kernels_attention_fp8_decode_gscores(stream: *mut c_void, q: *const u16, k_fp8: *const u8, v_fp8: *const u8, k_scales: *const f32, v_scales: *const f32, out: *mut u16, n_q: i32, n_kv: i32, head_dim: i32, n_total_dev: *const i32, max_total: i32, sliding_window: i32, scaling: f32, scores_gmem: *mut f32);
        kv_ring_append_bf16 = nv_kernels_kv_ring_append_bf16(stream: *mut c_void, src: *const u16, dst: *mut u16, pos_dev: *const i32, t: i32, cap: i32, n_kv: i32, head_dim: i32);
        kv_shift_bf16 = nv_kernels_kv_shift_bf16(stream: *mut c_void, buf: *mut u16, src_row: i32, dst_row: i32, rows: i32, n_kv: i32, head_dim: i32);
        attention_bf16_decode_ring = nv_kernels_attention_bf16_decode_ring(stream: *mut c_void, q: *const u16, k: *const u16, v: *const u16, out: *mut u16, ring_meta: *const i32, cap: i32, window: i32, n_q: i32, n_kv: i32, head_dim: i32, scaling: f32);
        sampler_topk_topp = nv_kernels_sampler_topk_topp(stream: *mut c_void, logits: *const f32, seeds: *const u64, probs_out: *mut f32, token_out: *mut u32, batch: usize, vocab: usize, temperature: f32, top_k: u32, top_p: f32);
        depthwise_conv1d_silu_bf16 = nv_kernels_depthwise_conv1d_silu_bf16(stream: *mut c_void, x_bf16: *const u16, w_bf16: *const u16, y_bf16: *mut u16, b: i32, c: i32, t: i32, k: i32);
        gather_rows_bf16 = nv_kernels_gather_rows_bf16(stream: *mut c_void, x_bf16: *const u16, src_idx: *const i32, out_bf16: *mut u16, m_total_padded: i32, hidden: i32, n_tokens: i32);
        copy_cols_bf16 = nv_kernels_copy_cols_bf16(stream: *mut c_void, src: *const u16, dst: *mut u16, rows: i32, width: i32, src_stride: i64, dst_stride: i64, src_off: i64, dst_off: i64);
        mul_sigmoid_rowgate_f32 = nv_kernels_mul_sigmoid_rowgate_f32(stream: *mut c_void, x: *const f32, gate_logits: *const f32, y: *mut f32, rows: i32, hidden: i32);
        moe_unpermute_scatter = nv_kernels_moe_unpermute_scatter(stream: *mut c_void, y_sorted_bf16: *const u16, topk_weights: *const f32, inv_perm: *const i32, y_acc_f32: *mut f32, n_tokens: i32, k: i32, hidden: i32, y_sorted_row_stride: i32);
        moe_unpermute_scatter_tail = nv_kernels_moe_unpermute_scatter_tail(stream: *mut c_void, y_sorted_bf16: *const u16, topk_weights: *const f32, inv_perm: *const i32, shared_f32: *const f32, resid_bf16: *const u16, out_bf16: *mut u16, n_tokens: i32, k: i32, hidden: i32, y_sorted_row_stride: i32);
        moe_gemv_swiglu_bf16_m1 = nv_kernels_moe_gemv_swiglu_bf16_m1(stream: *mut c_void, gate: *const u16, up: *const u16, ids: *const i32, x: *const u16, h: *mut u16, k: i32, num_experts: i32, inter: i32, hidden: i32);
        moe_gemv_swiglu_bf16_mb = nv_kernels_moe_gemv_swiglu_bf16_mb(stream: *mut c_void, gate: *const u16, up: *const u16, ids: *const i32, x: *const u16, h: *mut u16, b: i32, k: i32, num_experts: i32, inter: i32, hidden: i32);
        moe_gemv_down_tail_bf16_m1 = nv_kernels_moe_gemv_down_tail_bf16_m1(stream: *mut c_void, down: *const u16, ids: *const i32, weights: *const f32, h: *const u16, shared_f32: *const f32, resid: *const u16, out: *mut u16, k: i32, num_experts: i32, hidden: i32, inter: i32);
        moe_gemv_down_tail_bf16_mb = nv_kernels_moe_gemv_down_tail_bf16_mb(stream: *mut c_void, down: *const u16, ids: *const i32, weights: *const f32, h: *const u16, shared_f32: *const f32, resid: *const u16, out: *mut u16, b: i32, k: i32, num_experts: i32, hidden: i32, inter: i32);
        moe_permute = nv_kernels_moe_permute(stream: *mut c_void, topk_ids: *const i32, permuted_token_idx: *mut i32, expert_offsets: *mut i32, inv_perm: *mut i32, scratch_counts: *mut i32, n_tokens: i32, k: i32, num_experts: i32);
        moe_route_topk = nv_kernels_moe_route_topk(stream: *mut c_void, logits: *const f32, bias: *const f32, topk_ids: *mut i32, topk_weights: *mut f32, n_tokens: i32, num_experts: i32, k: i32, mode: i32, softcap: f32, norm_topk: i32, routed_scaling: f32);
        moe_route_topk_shared_tail = nv_kernels_moe_route_topk_shared_tail(stream: *mut c_void, logits: *const f32, bias: *const f32, topk_ids: *mut i32, topk_weights: *mut f32, n_tokens: i32, num_experts: i32, k: i32, mode: i32, softcap: f32, norm_topk: i32, routed_scaling: f32, shared_tail_id: i32);
        moe_route_gather_quant_m1 = nv_kernels_moe_route_gather_quant_m1(stream: *mut c_void, logits: *const f32, bias: *const f32, x_bf16: *const u16, globals_gu: *const f32, globals_dn: *const f32, topk_ids: *mut i32, topk_weights: *mut f32, gu_mini: *mut f32, dn_mini: *mut f32, x_fp4: *mut u8, x_sf: *mut u8, num_experts: i32, k: i32, mode: i32, softcap: f32, norm_topk: i32, routed_scaling: f32, shared_tail_id: i32, hidden: i32, min_tile: i32);
        gather_f32_by_ids = nv_kernels_gather_f32_by_ids(stream: *mut c_void, src: *const f32, ids: *const i32, dst: *mut f32, n: i32);
        dflash_accept_f32 = nv_kernels_dflash_accept_f32(stream: *mut c_void, logits: *const f32, drafts: *const u32, row_argmax: *mut u32, out: *mut u32, part_val: *mut f32, part_idx: *mut i32, m: i32, vocab: i32);
        softplus_gate_bf16 = nv_kernels_softplus_gate_bf16(stream: *mut c_void, attn: *const u16, gate: *const u16, out: *mut u16, groups: i32, hd: i32);
        softplus_gate_exact_bf16 = nv_kernels_softplus_gate_exact_bf16(stream: *mut c_void, attn: *const u16, gate: *const u16, out: *mut u16, groups: i32, hd: i32);
        laguna_rope_scale_bf16 = nv_kernels_laguna_rope_scale_bf16(stream: *mut c_void, q_in: *const u16, k_in: *const u16, q_out: *mut u16, k_out: *mut u16, cos_tbl: *const f32, sin_tbl: *const f32, pos_base: *const i32, t: i32, n_q: i32, n_kv: i32, head_dim: i32, rotary_dim: i32, rot_scale: f32);
        gemv_bf16_qkvg_normed = nv_kernels_gemv_bf16_qkvg_normed(stream: *mut c_void, wq: *const u16, wk: *const u16, wv: *const u16, wg: *const u16, x: *const u16, wn: *const u16, rstd: *const f32, yq: *mut u16, yk: *mut u16, yv: *mut u16, yg: *mut u16, nq: i32, nk: i32, nv: i32, ng: i32, k: i32);
        gemv_q8_qkvg_normed = nv_kernels_gemv_q8_qkvg_normed(stream: *mut c_void, fp8: i32, wq: *const c_void, sq: *const f32, wk: *const c_void, sk: *const f32, wv: *const c_void, sv: *const f32, wg: *const u16, x: *const u16, wn: *const u16, rstd: *const f32, yq: *mut u16, yk: *mut u16, yv: *mut u16, yg: *mut u16, nq: i32, nk: i32, nv: i32, ng: i32, k: i32);
        laguna_rstd256_bf16 = nv_kernels_laguna_rstd256_bf16(stream: *mut c_void, x: *const u16, rstd_out: *mut f32, dim: i32, eps: f32);
        laguna_qk_normrope_bf16 = nv_kernels_laguna_qk_normrope_bf16(stream: *mut c_void, q_in: *const u16, k_in: *const u16, q_out: *mut u16, k_out: *mut u16, qw: *const u16, kw: *const u16, cos_tbl: *const f32, sin_tbl: *const f32, pos_base: *const i32, n_q: i32, n_kv: i32, head_dim: i32, rotary_dim: i32, rot_scale: f32, eps_q: f32, eps_k: f32);
        laguna_flash_decode_gqa = nv_kernels_laguna_flash_decode_gqa(stream: *mut c_void, q: *const u16, k: *const u16, v: *const u16, out: *mut u16, total_ptr: *const i32, delta: i32, scratch: *mut f32, fan_in: *mut u32, n_q: i32, n_kv: i32, head_dim: i32, window: i32, scale: f32);
        laguna_seqlens_prep = nv_kernels_laguna_seqlens_prep(stream: *mut c_void, meta: *const i32, cu_full: *mut i32, cu_slide: *mut i32, t: i32);
        prof_timestamp = nv_kernels_prof_timestamp(stream: *mut c_void, out: *mut u64);
        silu_mul_quantize_nvfp4_bf16_per_expert = nv_kernels_silu_mul_quantize_nvfp4_bf16_per_expert(stream: *mut c_void, y_gate_bf16: *const u16, y_up_bf16: *const u16, packed_out: *mut u8, scales_out_swizzled: *mut u8, stored_globals: *const f32, m_per_expert: i32, m_total: i32, k: i32);
        quantize_nvfp4_bf16_per_expert = nv_kernels_quantize_nvfp4_bf16_per_expert(stream: *mut c_void, x_bf16: *const u16, packed_out: *mut u8, scales_out_swizzled: *mut u8, stored_globals: *const f32, m_per_expert: i32, m_total: i32, k: i32);
        silu_mul_quantize_nvfp4_bf16_per_expert_strided = nv_kernels_silu_mul_quantize_nvfp4_bf16_per_expert_strided(stream: *mut c_void, y_gate_bf16: *const u16, y_up_bf16: *const u16, packed_out: *mut u8, scales_out_swizzled: *mut u8, stored_globals: *const f32, m_per_expert: i32, m_total: i32, k: i32);
        quantize_nvfp4_bf16_per_expert_strided = nv_kernels_quantize_nvfp4_bf16_per_expert_strided(stream: *mut c_void, x_bf16: *const u16, packed_out: *mut u8, scales_out_swizzled: *mut u8, stored_globals: *const f32, m_per_expert: i32, m_total: i32, k: i32);
        gather_rows_bf16_strided = nv_kernels_gather_rows_bf16_strided(stream: *mut c_void, x_bf16: *const u16, src_idx: *const i32, out_bf16: *mut u16, m_total_padded: i32, hidden: i32, n_tokens: i32, row_stride: i32);
        quantize_nvfp4_bf16 = nv_kernels_quantize_nvfp4_bf16(stream: *mut c_void, x_bf16: *const u16, packed_out: *mut u8, scales_out_swizzled: *mut u8, stored_global: f32, m_padded: i32, m_logical: i32, k: i32);
        moe_grouped_fp4_gemv_m1_bf16 = nv_kernels_moe_grouped_fp4_gemv_m1_bf16(stream: *mut c_void, a_packed: *const u8, a_scales: *const u8, b_packed: *const u8, b_scales: *const u8, alphas: *const f32, d_bf16: *mut u16, group_expert_ids: *const i32, num_groups: i32, num_experts_total: i32, n: i32, k: i32, a_tile_stride_rows: i32, d_group_stride_elems: i64);
        gdn_recurrent_f32 = nv_kernels_gdn_recurrent_f32(stream: *mut c_void, q: *const f32, k: *const f32, v: *const f32, g_exp: *const f32, beta: *const f32, out: *mut f32, b: i32, t: i32, h: i32, k_dim: i32, v_dim: i32);
        gdn_prefill_qk_l2norm_from_mixed = nv_kernels_gdn_prefill_qk_l2norm_from_mixed(stream: *mut c_void, mixed: *const u16, q_out: *mut f32, k_out: *mut f32, bt: i32, hk: i32, conv_dim: i32, key_dim: i32, q_scale: f32, l2_eps: f32);
        gdn_recurrent_stateful_gqa_bf16out = nv_kernels_gdn_recurrent_stateful_gqa_bf16out(stream: *mut c_void, qn: *const f32, kn: *const f32, mixed: *const u16, g_exp: *const f32, beta: *const f32, state_inout: *mut f32, out: *mut u16, b: i32, t: i32, h: i32, hk: i32, k_dim: i32, v_dim: i32, conv_dim: i32, v_channel_offset: i32);
        gdn_conv1d_silu_bt_bf16 = nv_kernels_gdn_conv1d_silu_bt_bf16(stream: *mut c_void, x: *const u16, state_in: *const u16, w: *const u16, y: *mut u16, state_out: *mut u16, b: i32, t: i32, c: i32, k: i32);
        gdn_prefill_rmsnorm_gate_bf16 = nv_kernels_gdn_prefill_rmsnorm_gate_bf16(stream: *mut c_void, core: *const u16, z: *const u16, norm_weight: *const u16, gated: *mut u16, rows: i32, v_dim: i32, rms_eps: f32);
        gdn_conv_decode_silu_bf16 = nv_kernels_gdn_conv_decode_silu_bf16(stream: *mut c_void, x_new: *const u16, conv_state: *mut u16, w: *const u16, y: *mut u16, conv_dim: i32, k: i32);
        gdn_decode_step_bf16 = nv_kernels_gdn_decode_step_bf16(stream: *mut c_void, mixed: *const u16, z: *const u16, a: *const u16, b: *const u16, a_log: *const u16, dt_bias: *const u16, norm_w: *const u16, state: *mut f32, out: *mut u16, n_k: i32, n_v: i32, d_k: i32, d_v: i32, rms_eps: f32);
        gdn_decode_step_split_bf16 = nv_kernels_gdn_decode_step_split_bf16(stream: *mut c_void, mixed: *const u16, z: *const u16, a: *const u16, b: *const u16, a_log: *const u16, dt_bias: *const u16, norm_w: *const u16, state: *mut f32, out: *mut u16, qn_scratch: *mut f32, kn_scratch: *mut f32, g_exp_scratch: *mut f32, beta_scratch: *mut f32, core_scratch: *mut u16, n_k: i32, n_v: i32, d_k: i32, d_v: i32, rms_eps: f32);
        rmsnorm_residual_writeout_bf16 = nv_kernels_rmsnorm_residual_writeout_bf16(stream: *mut c_void, x: *const u16, res_in: *const u16, weight: *const u16, res_out: *mut u16, out: *mut u16, batch: usize, hidden: usize, eps: f32);
        gdn_decode_step_ab_fused_bf16 = nv_kernels_gdn_decode_step_ab_fused_bf16(stream: *mut c_void, x: *const u16, a_w: *const u16, b_w: *const u16, mixed: *const u16, z: *const u16, a_log: *const u16, dt_bias: *const u16, norm_w: *const u16, state: *mut f32, out: *mut u16, hidden: i32, n_k: i32, n_v: i32, d_k: i32, d_v: i32, rms_eps: f32);
        gemv_e4m3_qkvz_conv_m1 = nv_kernels_gemv_e4m3_qkvz_conv_m1(stream: *mut c_void, wq: *const u8, row_scale: *const f32, x: *const u16, conv_w: *const u16, conv_state: *mut u16, mixed_out: *mut u16, z_out: *mut u16, n: i32, k: i32, conv_dim: i32, k_c: i32);
        gdn_conv_decode_chunk_silu_bf16 = nv_kernels_gdn_conv_decode_chunk_silu_bf16(stream: *mut c_void, x_seq: *const u16, conv_state: *const u16, w: *const u16, y: *mut u16, ckpt_conv: *mut u16, conv_dim: i32, k: i32, t: i32);
        gdn_decode_chunk_bf16 = nv_kernels_gdn_decode_chunk_bf16(stream: *mut c_void, mixed: *const u16, z: *const u16, a: *const u16, b: *const u16, a_log: *const u16, dt_bias: *const u16, norm_w: *const u16, state_in: *const f32, ckpt_state: *mut f32, out: *mut u16, n_k: i32, n_v: i32, d_k: i32, d_v: i32, rms_eps: f32, t: i32);
        gdn_decode_chunk_split_bf16 = nv_kernels_gdn_decode_chunk_split_bf16(stream: *mut c_void, mixed: *const u16, z: *const u16, a: *const u16, b: *const u16, a_log: *const u16, dt_bias: *const u16, norm_w: *const u16, state_in: *const f32, ckpt_state: *mut f32, live_state_out: *mut f32, out: *mut u16, qn_scratch: *mut f32, kn_scratch: *mut f32, g_exp_scratch: *mut f32, beta_scratch: *mut f32, core_scratch: *mut u16, n_k: i32, n_v: i32, d_k: i32, d_v: i32, rms_eps: f32, t: i32);
        rmsnorm_residual_bf16 = nv_kernels_rmsnorm_residual_bf16(stream: *mut c_void, x: *const u16, residual: *mut u16, weight: *const u16, out: *mut u16, batch: usize, hidden: usize, eps: f32);
        rmsnorm_residual_f32 = nv_kernels_rmsnorm_residual_f32(stream: *mut c_void, x: *const f32, residual: *mut f32, weight: *const f32, out: *mut f32, batch: usize, hidden: usize, eps: f32);
        gdn_gating_bf16 = nv_kernels_gdn_gating_bf16(stream: *mut c_void, a: *const u16, b: *const u16, a_log: *const u16, dt_bias: *const u16, g_out: *mut f32, beta_out: *mut u16, tokens: usize, num_heads: usize);
        gdn_gating_f32 = nv_kernels_gdn_gating_f32(stream: *mut c_void, a: *const f32, b: *const f32, a_log: *const f32, dt_bias: *const f32, g_out: *mut f32, beta_out: *mut f32, tokens: usize, num_heads: usize);
        gemv_w4a16_m1_proto = nv_kernels_gemv_w4a16_m1_proto(stream: *mut c_void, packed: *const u32, scale: *const u16, x: *const u16, y: *mut u16, n: i32, k: i32, gs: i32, variant: i32);
        argmax_f32_rows = nv_kernels_argmax_f32_rows(stream: *mut c_void, logits: *const f32, rows: i32, n: i32, part_val: *mut f32, part_idx: *mut i32, out: *mut u32);
        flash_decode_fused_bf16kv_mk = nv_kernels_flash_decode_fused_bf16kv_mk(stream: *mut c_void, q: *const f32, k: *const u16, v: *const u16, out: *mut u16, pos: *const i32, delta: i32, m: i32, scratch: *mut f32, fan_in: *mut u32, nh: i32, nkv: i32, hd: i32, window: i32);
        flash_decode_gqa_fp8kv_paged = nv_kernels_flash_decode_gqa_fp8kv_paged(stream: *mut c_void, q: *const u16, k_fp8: *const u8, v_fp8: *const u8, k_scales: *const f32, v_scales: *const f32, out: *mut u16, n_total_dev: *const i32, scratch: *mut f32, fan_in: *mut u32, nh: i32, nkv: i32, hd: i32, window: i32, ring: i32, splits: i32, scaling: f32, block_table: *const i32, block_size: i32);
        flash_decode_kvshare_fp8kv_paged = nv_kernels_flash_decode_kvshare_fp8kv_paged(stream: *mut c_void, q: *const u16, k_fp8: *const u8, v_fp8: *const u8, k_scales: *const f32, v_scales: *const f32, out: *mut u16, n_total_dev: *const i32, scratch: *mut f32, fan_in: *mut u32, nh: i32, nkv: i32, hd: i32, window: i32, ring: i32, splits: i32, scaling: f32, block_table: *const i32, block_size: i32);
        kvshare_bw_probe = nv_kernels_kvshare_bw_probe(stream: *mut c_void, q: *const u16, k_fp8: *const u8, v_fp8: *const u8, k_scales: *const f32, v_scales: *const f32, block_table: *const i32, block_size: i32, sink: *mut f32, total: i32, nkv: i32, splits: i32, mode: i32);
        flash_decode_derivev_fp8kv_paged = nv_kernels_flash_decode_derivev_fp8kv_paged(stream: *mut c_void, q: *const u16, k_fp8: *const u8, k_scales: *const f32, inv_freq: *const f32, cos_pk: *const f32, sin_pk: *const f32, out: *mut u16, n_total_dev: *const i32, scratch: *mut f32, fan_in: *mut u32, nh: i32, nkv: i32, hd: i32, window: i32, ring: i32, rope_angles: i32, w_inv: f32, scaling: f32, block_table: *const i32, block_size: i32);
        flash_decode_fused_fp8kv_paged = nv_kernels_flash_decode_fused_fp8kv_paged(stream: *mut c_void, q: *const u16, k_fp8: *const u8, v_fp8: *const u8, k_scales: *const f32, v_scales: *const f32, out: *mut u16, n_total_dev: *const i32, scratch: *mut f32, fan_in: *mut u32, nh: i32, nkv: i32, hd: i32, window: i32, ring: i32, scaling: f32, block_table: *const i32, block_size: i32);
        flash_decode_fused_fp8kv_mk_paged = nv_kernels_flash_decode_fused_fp8kv_mk_paged(stream: *mut c_void, q: *const u16, k_fp8: *const u8, v_fp8: *const u8, k_scales: *const f32, v_scales: *const f32, out: *mut u16, n_total_dev: *const i32, delta: i32, m: i32, scratch: *mut f32, fan_in: *mut u32, nh: i32, nkv: i32, hd: i32, window: i32, ring: i32, scaling: f32, block_table: *const i32, block_size: i32);
        flash_decode_fused_fp8kv = nv_kernels_flash_decode_fused_fp8kv(stream: *mut c_void, q: *const u16, k_fp8: *const u8, v_fp8: *const u8, k_scales: *const f32, v_scales: *const f32, out: *mut u16, n_total_dev: *const i32, scratch: *mut f32, fan_in: *mut u32, nh: i32, nkv: i32, hd: i32, window: i32, ring: i32, scaling: f32);
        flash_decode_fused_fp8kv_mk = nv_kernels_flash_decode_fused_fp8kv_mk(stream: *mut c_void, q: *const u16, k_fp8: *const u8, v_fp8: *const u8, k_scales: *const f32, v_scales: *const f32, out: *mut u16, n_total_dev: *const i32, delta: i32, m: i32, scratch: *mut f32, fan_in: *mut u32, nh: i32, nkv: i32, hd: i32, window: i32, ring: i32, scaling: f32);
        rope_bf16_oop = nv_kernels_rope_bf16_oop(stream: *mut c_void, q_in: *const u16, k_in: *const u16, q_out: *mut u16, k_out: *mut u16, cos_tbl: *const f32, sin_tbl: *const f32, positions: *const i32, batch: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize);
        token_map_u32 = nv_kernels_token_map_u32(stream: *mut c_void, map: *const u32, idx: *const u32, out: *mut u32);
    }

    pub fn capability() -> Result<(i32, i32), i32> {
        let mut major = 0i32;
        let mut minor = 0i32;
        let rc = unsafe { sys::nv_kernels_capability(&mut major, &mut minor) };
        if rc == 0 {
            Ok((major, minor))
        } else {
            Err(rc)
        }
    }

    pub fn argmax_parts() -> usize {
        (unsafe { sys::nv_kernels_argmax_bf16_parts() }) as usize
    }

    pub fn flash_splitk_scratch_elems(nh: i32, hd: i32) -> usize {
        (unsafe { sys::nv_kernels_flash_splitk_scratch_elems(nh, hd) }) as usize
    }

    pub fn gqa512_scratch_elems(nh: i32, m: i32, splits: i32) -> usize {
        (unsafe { sys::nv_kernels_gqa512_scratch_elems(nh, m, splits) }) as usize
    }

    pub fn gemv_i8_normed_mk_max_m(k: i32) -> i32 {
        unsafe { sys::nv_kernels_gemv_i8_normed_mk_max_m(k) }
    }

    pub fn silu_mul_rowquant_i8_mk_partials_len(m: i32, k: i32) -> i32 {
        unsafe { sys::nv_kernels_silu_mul_rowquant_i8_mk_partials_len(m, k) }
    }

    pub fn cutlass_flashinfer_probe() -> Result<(i32, i32), i32> {
        let mut status = -1i32;
        let mut maxx32 = -1i32;
        let rc = unsafe { sys::nv_kernels_cutlass_flashinfer_probe(&mut status, &mut maxx32) };
        if rc == 0 {
            Ok((status, maxx32))
        } else {
            Err(rc)
        }
    }

    pub fn dflash_accept_parts() -> usize {
        (unsafe { sys::nv_kernels_dflash_accept_parts() }) as usize
    }

    pub fn laguna_flash_decode_gqa_scratch_elems(n_kv: i32) -> usize {
        (unsafe { sys::nv_kernels_laguna_flash_decode_gqa_scratch_elems(n_kv) }) as usize
    }

    #[cfg(feature = "cuda")]
    pub unsafe fn quantize_nvfp4_bf16_rows(
        stream: *mut c_void,
        x_bf16: *const u16,
        packed_out: *mut u8,
        scales_out_swizzled: *mut u8,
        stored_global: f32,
        m_rows: i32,
        k: i32,
    ) -> i32 {
        sys::nv_kernels_quantize_nvfp4_bf16_rows(
            stream,
            x_bf16,
            packed_out,
            scales_out_swizzled,
            stored_global,
            m_rows,
            k,
        )
    }

    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    pub unsafe fn quantize_nvfp4_bf16_rows(
        stream: *mut c_void,
        x_bf16: *const u16,
        packed_out: *mut u8,
        scales_out_swizzled: *mut u8,
        stored_global: f32,
        m_rows: i32,
        k: i32,
    ) -> i32 {
        sys::nv_kernels_quantize_nvfp4_bf16(
            stream,
            x_bf16,
            packed_out,
            scales_out_swizzled,
            stored_global,
            m_rows,
            m_rows,
            k,
        )
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rmsnorm_quantize_nvfp4_bf16(
        stream: *mut c_void,
        x_bf16: *const u16,
        weight_bf16: *const u16,
        packed_out: *mut u8,
        scales_out_swizzled: *mut u8,
        stored_global: f32,
        eps: f32,
        m_padded: i32,
        m_logical: i32,
        k: i32,
    ) -> i32 {
        sys::nv_kernels_rmsnorm_quantize_nvfp4_bf16(
            stream,
            x_bf16,
            weight_bf16,
            packed_out,
            scales_out_swizzled,
            stored_global,
            eps,
            m_padded,
            m_logical,
            k,
        )
    }

    pub unsafe fn cutlass_moe_grouped_fp4_gemm_sm120_bf16(
        stream: *mut c_void,
        a_packed: *const c_void,
        a_scales: *const c_void,
        b_packed: *const c_void,
        b_scales: *const c_void,
        alphas: *const f32,
        d_bf16: *mut c_void,
        expert_offsets: *const i32,
        sf_offsets: *const i32,
        problem_sizes: *const i32,
        active_expert_indices: *const i32,
        n: i32,
        k: i32,
        num_experts: i32,
        a_row_stride_elems: i64,
        b_row_stride_elems: i64,
        c_row_stride_elems: i64,
        meta_scratch: *mut c_void,
        meta_scratch_bytes: usize,
        gemm_workspace: *mut c_void,
        gemm_workspace_bytes: usize,
    ) -> Result<usize, i32> {
        let mut needed: usize = 0;
        let rc = sys::nv_kernels_cutlass_moe_grouped_fp4_gemm_sm120_bf16(
            stream,
            a_packed,
            a_scales,
            b_packed,
            b_scales,
            alphas,
            d_bf16,
            expert_offsets,
            sf_offsets,
            problem_sizes,
            active_expert_indices,
            n,
            k,
            num_experts,
            a_row_stride_elems,
            b_row_stride_elems,
            c_row_stride_elems,
            meta_scratch,
            meta_scratch_bytes,
            gemm_workspace,
            gemm_workspace_bytes,
            &mut needed,
        );
        if rc == 0 {
            Ok(needed)
        } else {
            Err(rc)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cutlass_moe_grouped_fp4_gemm_sm120_bf16_decode(
        stream: *mut c_void,
        a_packed: *const c_void,
        a_scales: *const c_void,
        b_packed: *const c_void,
        b_scales: *const c_void,
        alphas: *const f32,
        d_bf16: *mut c_void,
        expert_offsets: *const i32,
        sf_offsets: *const i32,
        problem_sizes: *const i32,
        active_expert_indices: *const i32,
        n: i32,
        k: i32,
        num_experts: i32,
        a_row_stride_elems: i64,
        b_row_stride_elems: i64,
        c_row_stride_elems: i64,
        meta_scratch: *mut c_void,
        meta_scratch_bytes: usize,
        gemm_workspace: *mut c_void,
        gemm_workspace_bytes: usize,
    ) -> Result<usize, i32> {
        let mut needed: usize = 0;
        let rc = sys::nv_kernels_cutlass_moe_grouped_fp4_gemm_sm120_bf16_decode(
            stream,
            a_packed,
            a_scales,
            b_packed,
            b_scales,
            alphas,
            d_bf16,
            expert_offsets,
            sf_offsets,
            problem_sizes,
            active_expert_indices,
            n,
            k,
            num_experts,
            a_row_stride_elems,
            b_row_stride_elems,
            c_row_stride_elems,
            meta_scratch,
            meta_scratch_bytes,
            gemm_workspace,
            gemm_workspace_bytes,
            &mut needed,
        );
        if rc == 0 {
            Ok(needed)
        } else {
            Err(rc)
        }
    }

    pub unsafe fn cutlass_fp4_gemm_sm120_bf16(
        stream: *mut c_void,
        a_fp4: *const c_void,
        a_sf: *const c_void,
        b_fp4: *const c_void,
        b_sf: *const c_void,
        global_sf: *const f32,
        d_bf16: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
    ) -> Result<usize, i32> {
        let mut needed: usize = 0;
        let rc = sys::nv_kernels_cutlass_fp4_gemm_sm120_bf16(
            stream,
            a_fp4,
            a_sf,
            b_fp4,
            b_sf,
            global_sf,
            d_bf16,
            m,
            n,
            k,
            workspace,
            workspace_bytes,
            &mut needed,
        );
        if rc == 0 {
            Ok(needed)
        } else {
            Err(rc)
        }
    }

    pub unsafe fn cutlass_fp4_gemm_sm120_bf16_streamk(
        stream: *mut c_void,
        a_fp4: *const c_void,
        a_sf: *const c_void,
        b_fp4: *const c_void,
        b_sf: *const c_void,
        global_sf: *const f32,
        d_bf16: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
    ) -> Result<usize, i32> {
        let mut needed: usize = 0;
        let rc = sys::nv_kernels_cutlass_fp4_gemm_sm120_bf16_streamk(
            stream,
            a_fp4,
            a_sf,
            b_fp4,
            b_sf,
            global_sf,
            d_bf16,
            m,
            n,
            k,
            workspace,
            workspace_bytes,
            &mut needed,
        );
        if rc == 0 {
            Ok(needed)
        } else {
            Err(rc)
        }
    }

    pub unsafe fn cutlass_fp4_gemm_sm120_bf16_tiled(
        stream: *mut c_void,
        a_fp4: *const c_void,
        a_sf: *const c_void,
        b_fp4: *const c_void,
        b_sf: *const c_void,
        global_sf: *const f32,
        d_bf16: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        tile: i32,
        stream_k: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
    ) -> Result<usize, i32> {
        let mut needed: usize = 0;
        let rc = sys::nv_kernels_cutlass_fp4_gemm_sm120_bf16_tiled(
            stream,
            a_fp4,
            a_sf,
            b_fp4,
            b_sf,
            global_sf,
            d_bf16,
            m,
            n,
            k,
            tile,
            stream_k,
            workspace,
            workspace_bytes,
            &mut needed,
        );
        if rc == 0 {
            Ok(needed)
        } else {
            Err(rc)
        }
    }

    pub fn flash_splitk_scratch_elems_mk(nh: i32, hd: i32, m: i32) -> usize {
        (unsafe { sys::nv_kernels_flash_splitk_scratch_elems_mk(nh, hd, m) }) as usize
    }

}

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
pub mod cuda {
    use std::ffi::c_void;

    const RC_BUILT_WITHOUT_GPU_BACKEND: i32 = -1;

    macro_rules! launchers_report_no_gpu_backend {
        ($($name:ident ( $($ty:ty),* );)*) => {
            $(pub unsafe fn $name($(_: $ty),*) -> i32 {
                RC_BUILT_WITHOUT_GPU_BACKEND
            })*
        };
    }

    launchers_report_no_gpu_backend! {
        hello_launch(*mut c_void, *mut f32, usize);
        rmsnorm_f32(*mut c_void, *const f32, *const f32, *mut f32, usize, usize, f32);
        rmsnorm_bf16(*mut c_void, *const u16, *const u16, *mut u16, usize, usize, f32);
        rope_f32(*mut c_void, *mut f32, *mut f32, *const f32, *const f32, *const i32, usize, usize, usize, usize);
        rope_bf16(*mut c_void, *mut u16, *mut u16, *const f32, *const f32, *const i32, usize, usize, usize, usize);
        silu_f32(*mut c_void, *const f32, *mut f32, usize);
        silu_bf16(*mut c_void, *const u16, *mut u16, usize);
        silu_mul_f32(*mut c_void, *const f32, *const f32, *mut f32, usize);
        silu_mul_bf16(*mut c_void, *const u16, *const u16, *mut u16, usize);
        silu_mul_bf16_candle(*mut c_void, *const u16, *const u16, *mut u16, usize);
        gelu_tanh_mul_bf16(*mut c_void, *const u16, *const u16, *mut u16, usize);
        gemv_bf16(*mut c_void, *const u16, *const u16, *mut u16, i32, i32);
        gemm_bf16_mk(*mut c_void, *const u16, *const u16, *mut u16, i32, i32, i32);
        nvfp4_quantize_row_bf16(*mut c_void, *const u16, *mut u8, *mut u8, f32, i32);
        nvfp4_gemv_bf16(*mut c_void, *const u8, *const u8, *const u8, *const u8, *mut u16, f32, i32, i32);
        quantize_kv_fp8(*mut c_void, *const u16, *mut u8, *mut f32, *const i32, i32, i32, i32, i32);
        dequantize_kv_fp8(*mut c_void, *const u8, *const f32, *mut u16, i32, i32, i32, i32, i32);
        quantize_kv_fp8_paged(*mut c_void, *const u16, *mut u8, *mut f32, *const i32, *const i32, i32, i32, i32, i32);
        dequantize_kv_fp8_paged(*mut c_void, *const u8, *const f32, *mut u16, *const i32, i32, i32, i32, i32);
        derive_v_from_k_fp8_paged(*mut c_void, *const u8, *const f32, *const f32, *const f32, *const f32, *mut u16, *const i32, i32, i32, i32, i32, i32, i32, i32, f32);
        copy_kv_block_fp8(*mut c_void, *const u8, *const f32, *mut u8, *mut f32, i32, i32, i32, i32, i32);
        kvshare_bw_probe(*mut c_void, *const u16, *const u8, *const u8, *const f32, *const f32, *const i32, i32, *mut f32, i32, i32, i32, i32);
        attention_fp8_decode(*mut c_void, *const u16, *const u8, *const u8, *const f32, *const f32, *mut u16, i32, i32, i32, *const i32, i32, i32, f32);
        attention_fp8_decode_gscores(*mut c_void, *const u16, *const u8, *const u8, *const f32, *const f32, *mut u16, i32, i32, i32, *const i32, i32, i32, f32, *mut f32);
        kv_ring_append_bf16(*mut c_void, *const u16, *mut u16, *const i32, i32, i32, i32, i32);
        kv_shift_bf16(*mut c_void, *mut u16, i32, i32, i32, i32, i32);
        attention_bf16_decode_ring(*mut c_void, *const u16, *const u16, *const u16, *mut u16, *const i32, i32, i32, i32, i32, i32, f32);
        sampler_topk_topp(*mut c_void, *const f32, *const u64, *mut f32, *mut u32, usize, usize, f32, u32, f32);
        moe_route_topk(*mut c_void, *const f32, *const f32, *mut i32, *mut f32, i32, i32, i32, i32, f32, i32, f32);
        moe_route_topk_shared_tail(*mut c_void, *const f32, *const f32, *mut i32, *mut f32, i32, i32, i32, i32, f32, i32, f32, i32);
        moe_route_gather_quant_m1(*mut c_void, *const f32, *const f32, *const u16, *const f32, *const f32, *mut i32, *mut f32, *mut f32, *mut f32, *mut u8, *mut u8, i32, i32, i32, f32, i32, f32, i32, i32, i32);
        gather_f32_by_ids(*mut c_void, *const f32, *const i32, *mut f32, i32);
        dflash_accept_f32(*mut c_void, *const f32, *const u32, *mut u32, *mut u32, *mut f32, *mut i32, i32, i32);
        quantize_nvfp4_bf16(*mut c_void, *const u16, *mut u8, *mut u8, f32, i32, i32, i32);
        quantize_nvfp4_bf16_per_expert(*mut c_void, *const u16, *mut u8, *mut u8, *const f32, i32, i32, i32);
        moe_permute(*mut c_void, *const i32, *mut i32, *mut i32, *mut i32, *mut i32, i32, i32, i32);
        moe_unpermute_scatter(*mut c_void, *const u16, *const f32, *const i32, *mut f32, i32, i32, i32, i32);
        moe_unpermute_scatter_tail(*mut c_void, *const u16, *const f32, *const i32, *const f32, *const u16, *mut u16, i32, i32, i32, i32);
        gather_rows_bf16(*mut c_void, *const u16, *const i32, *mut u16, i32, i32, i32);
        copy_cols_bf16(*mut c_void, *const u16, *mut u16, i32, i32, i64, i64, i64, i64);
        mul_sigmoid_rowgate_f32(*mut c_void, *const f32, *const f32, *mut f32, i32, i32);
        depthwise_conv1d_silu_bf16(*mut c_void, *const u16, *const u16, *mut u16, i32, i32, i32, i32);
        silu_mul_quantize_nvfp4_bf16_per_expert(*mut c_void, *const u16, *const u16, *mut u8, *mut u8, *const f32, i32, i32, i32);
        silu_mul_quantize_nvfp4_bf16_per_expert_strided(*mut c_void, *const u16, *const u16, *mut u8, *mut u8, *const f32, i32, i32, i32);
        quantize_nvfp4_bf16_per_expert_strided(*mut c_void, *const u16, *mut u8, *mut u8, *const f32, i32, i32, i32);
        gather_rows_bf16_strided(*mut c_void, *const u16, *const i32, *mut u16, i32, i32, i32, i32);
        gdn_recurrent_f32(*mut c_void, *const f32, *const f32, *const f32, *const f32, *const f32, *mut f32, i32, i32, i32, i32, i32);
        gdn_prefill_qk_l2norm_from_mixed(*mut c_void, *const u16, *mut f32, *mut f32, i32, i32, i32, i32, f32, f32);
        gdn_recurrent_stateful_gqa_bf16out(*mut c_void, *const f32, *const f32, *const u16, *const f32, *const f32, *mut f32, *mut u16, i32, i32, i32, i32, i32, i32, i32, i32);
        gdn_conv1d_silu_bt_bf16(*mut c_void, *const u16, *const u16, *const u16, *mut u16, *mut u16, i32, i32, i32, i32);
        gdn_prefill_rmsnorm_gate_bf16(*mut c_void, *const u16, *const u16, *const u16, *mut u16, i32, i32, f32);
        gdn_conv_decode_silu_bf16(*mut c_void, *const u16, *mut u16, *const u16, *mut u16, i32, i32);
        gdn_decode_step_bf16(*mut c_void, *const u16, *const u16, *const u16, *const u16, *const u16, *const u16, *const u16, *mut f32, *mut u16, i32, i32, i32, i32, f32);
        gdn_decode_step_split_bf16(*mut c_void, *const u16, *const u16, *const u16, *const u16, *const u16, *const u16, *const u16, *mut f32, *mut u16, *mut f32, *mut f32, *mut f32, *mut f32, *mut u16, i32, i32, i32, i32, f32);
        rmsnorm_residual_writeout_bf16(*mut c_void, *const u16, *const u16, *const u16, *mut u16, *mut u16, usize, usize, f32);
        gdn_decode_step_ab_fused_bf16(*mut c_void, *const u16, *const u16, *const u16, *const u16, *const u16, *const u16, *const u16, *const u16, *mut f32, *mut u16, i32, i32, i32, i32, i32, f32);
        gemv_e4m3_qkvz_conv_m1(*mut c_void, *const u8, *const f32, *const u16, *const u16, *mut u16, *mut u16, *mut u16, i32, i32, i32, i32);
        gdn_conv_decode_chunk_silu_bf16(*mut c_void, *const u16, *const u16, *const u16, *mut u16, *mut u16, i32, i32, i32);
        gdn_decode_chunk_bf16(*mut c_void, *const u16, *const u16, *const u16, *const u16, *const u16, *const u16, *const u16, *const f32, *mut f32, *mut u16, i32, i32, i32, i32, f32, i32);
        gdn_decode_chunk_split_bf16(*mut c_void, *const u16, *const u16, *const u16, *const u16, *const u16, *const u16, *const u16, *const f32, *mut f32, *mut f32, *mut u16, *mut f32, *mut f32, *mut f32, *mut f32, *mut u16, i32, i32, i32, i32, f32, i32);
        rmsnorm_residual_bf16(*mut c_void, *const u16, *mut u16, *const u16, *mut u16, usize, usize, f32);
        rmsnorm_residual_f32(*mut c_void, *const f32, *mut f32, *const f32, *mut f32, usize, usize, f32);
        gdn_gating_bf16(*mut c_void, *const u16, *const u16, *const u16, *const u16, *mut f32, *mut u16, usize, usize);
        gdn_gating_f32(*mut c_void, *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, usize, usize);
        rope_bf16_oop(*mut c_void, *const u16, *const u16, *mut u16, *mut u16, *const f32, *const f32, *const i32, usize, usize, usize, usize);
    }

    pub fn capability() -> Result<(i32, i32), i32> {
        Err(-1)
    }

    pub fn cutlass_flashinfer_probe() -> Result<(i32, i32), i32> {
        Err(-1)
    }

    pub fn dflash_accept_parts() -> usize {
        0
    }

    pub unsafe fn quantize_nvfp4_bf16_rows(
        _stream: *mut c_void,
        _x_bf16: *const u16,
        _packed_out: *mut u8,
        _scales_out_swizzled: *mut u8,
        _stored_global: f32,
        _m_rows: i32,
        _k: i32,
    ) -> i32 {
        -1
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rmsnorm_quantize_nvfp4_bf16(
        _stream: *mut c_void,
        _x_bf16: *const u16,
        _weight_bf16: *const u16,
        _packed_out: *mut u8,
        _scales_out_swizzled: *mut u8,
        _stored_global: f32,
        _eps: f32,
        _m_padded: i32,
        _m_logical: i32,
        _k: i32,
    ) -> i32 {
        -1
    }

    pub unsafe fn cutlass_fp4_gemm_sm120_bf16(
        _stream: *mut c_void,
        _a_fp4: *const c_void,
        _a_sf: *const c_void,
        _b_fp4: *const c_void,
        _b_sf: *const c_void,
        _global_sf: *const f32,
        _d_bf16: *mut c_void,
        _m: i32,
        _n: i32,
        _k: i32,
        _workspace: *mut c_void,
        _workspace_bytes: usize,
    ) -> Result<usize, i32> {
        Err(-1)
    }

    pub unsafe fn cutlass_fp4_gemm_sm120_bf16_streamk(
        _stream: *mut c_void,
        _a_fp4: *const c_void,
        _a_sf: *const c_void,
        _b_fp4: *const c_void,
        _b_sf: *const c_void,
        _global_sf: *const f32,
        _d_bf16: *mut c_void,
        _m: i32,
        _n: i32,
        _k: i32,
        _workspace: *mut c_void,
        _workspace_bytes: usize,
    ) -> Result<usize, i32> {
        Err(-1)
    }

    pub unsafe fn cutlass_fp4_gemm_sm120_bf16_tiled(
        _stream: *mut c_void,
        _a_fp4: *const c_void,
        _a_sf: *const c_void,
        _b_fp4: *const c_void,
        _b_sf: *const c_void,
        _global_sf: *const f32,
        _d_bf16: *mut c_void,
        _m: i32,
        _n: i32,
        _k: i32,
        _tile: i32,
        _stream_k: i32,
        _workspace: *mut c_void,
        _workspace_bytes: usize,
    ) -> Result<usize, i32> {
        Err(-1)
    }

}
