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

mod launcher_registry;

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

    pub fn nv_kernels_lora_m1_shrink_tiles(
        stream: *mut std::ffi::c_void,
        x_bf16: *const u16,
        a_ptrs: *const u64,
        partials: *mut f32,
        n_slices: i32,
        rank: i32,
        k: i32,
        k_tile: i32,
        n_tiles: i32,
    ) -> i32;

    pub fn nv_kernels_lora_m1_expand_tiles(
        stream: *mut std::ffi::c_void,
        partials: *const f32,
        b_ptrs: *const u64,
        y_bf16: *mut u16,
        slice_n: *const i32,
        slice_start: *const i32,
        n_slices: i32,
        rank: i32,
        n_tiles: i32,
        max_n: i32,
        win_off: i32,
        win_len: i32,
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

#[cfg(feature = "wgpu")]
#[path = "wgpu/mod.rs"]
pub mod wgpu_qlora;

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

    crate::launcher_registry::launcher_signatures_listed_once_so_the_no_gpu_stub_arm_cannot_drift!(
        launchers_forward_to_the_c_symbol_verbatim
    );

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

    pub fn flash_decode_gqa4_hd512_scratch_elems(nh: i32, capacity: i32) -> usize {
        (unsafe { sys::nv_kernels_flash_decode_gqa4_hd512_scratch_elems(nh, capacity) }) as usize
    }

    pub fn gdn_chunk_gemm_scan_workspace_elems(t: i32, h: i32, hk: i32) -> usize {
        (unsafe { sys::nv_kernels_gdn_chunk_gemm_scan_workspace_elems(t, h, hk) }) as usize
    }

    pub fn gdn_chunk_gemm_graph_replays_total() -> i64 {
        unsafe { sys::nv_kernels_gdn_chunk_gemm_graph_replays_total() }
    }

    pub fn gdn_prefill_tc_head_launches_total() -> i64 {
        unsafe { sys::nv_kernels_gdn_prefill_tc_head_launches_total() }
    }

    pub fn gdn_scan_tree_compose_floor_workspace_elems(n: i32, h: i32) -> usize {
        (unsafe { sys::nv_kernels_gdn_scan_tree_compose_floor_workspace_elems(n, h) }) as usize
    }

    pub unsafe fn gdn_scan_tree_compose_floor(
        stream: *mut std::ffi::c_void,
        workspace: *mut f32,
        workspace_elems: i64,
        n: i32,
        h: i32,
    ) -> i32 {
        sys::nv_kernels_gdn_scan_tree_compose_floor(stream, workspace, workspace_elems, n, h)
    }

    pub fn flash_depthgate_knee_tokens() -> usize {
        (unsafe { sys::nv_kernels_flash_depthgate_knee_tokens() }) as usize
    }

    pub fn flash_decode_fp8_splits_pick(nh: i32, window: i32) -> i32 {
        unsafe { sys::nv_kernels_flash_decode_fp8_splits_pick(nh, window) }
    }

    pub fn flash_decode_fp8_splits_pick_for_a_lane_group(
        nh: i32,
        window: i32,
        lanes_in_flight: i32,
    ) -> i32 {
        unsafe {
            sys::nv_kernels_flash_decode_fp8_splits_pick_for_a_lane_group(
                nh,
                window,
                lanes_in_flight,
            )
        }
    }

    pub fn flash_sm_count() -> i32 {
        unsafe { sys::nv_kernels_flash_sm_count() }
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

    pub unsafe fn nvfp4_gemm_sm120_v192_bf16(
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
    ) -> Result<(), i32> {
        let rc = sys::nv_kernels_nvfp4_gemm_sm120_v192_bf16(
            stream, a_fp4, a_sf, b_fp4, b_sf, global_sf, d_bf16, m, n, k,
        );
        if rc == 0 {
            Ok(())
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
        ($($name:ident = $sym:ident ( $($arg:ident : $ty:ty),* );)*) => {
            $(pub unsafe fn $name($(_: $ty),*) -> i32 {
                RC_BUILT_WITHOUT_GPU_BACKEND
            })*
        };
    }

    crate::launcher_registry::launcher_signatures_listed_once_so_the_no_gpu_stub_arm_cannot_drift!(
        launchers_report_no_gpu_backend
    );

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

    pub unsafe fn nvfp4_gemm_sm120_v192_bf16(
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
    ) -> Result<(), i32> {
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
