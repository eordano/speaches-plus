use std::ffi::c_void;

pub use crate::lora_meta::{LoraKernelMeta, NO_LORA};

pub const M1_MAX_RANK: usize = 64;
pub const M1_SHRINK_TILE_K_512_KEEPS_THE_PER_LANE_DRAM_LATENCY_CHAIN_UNDER_10_LOADS: usize = 512;
pub const M1_SHRINK_MAX_TILES: usize = 64;

pub fn m1_shrink_tiling(k: usize) -> (usize, usize) {
    let tiles = k
        .div_ceil(M1_SHRINK_TILE_K_512_KEEPS_THE_PER_LANE_DRAM_LATENCY_CHAIN_UNDER_10_LOADS)
        .clamp(1, M1_SHRINK_MAX_TILES);
    let k_tile = k.div_ceil(tiles).next_multiple_of(2);
    (tiles, k_tile)
}

pub unsafe fn lora_m1_shrink_tiles(
    stream: *mut c_void,
    x_bf16: *const u16,
    a_ptrs: *const u64,
    partials: *mut f32,
    n_slices: i32,
    rank: i32,
    k: i32,
    k_tile: i32,
    n_tiles: i32,
) -> i32 {
    unsafe {
        crate::nv_kernels_lora_m1_shrink_tiles(
            stream, x_bf16, a_ptrs, partials, n_slices, rank, k, k_tile, n_tiles,
        )
    }
}

pub unsafe fn lora_m1_expand_tiles(
    stream: *mut c_void,
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
) -> i32 {
    unsafe {
        crate::nv_kernels_lora_m1_expand_tiles(
            stream,
            partials,
            b_ptrs,
            y_bf16,
            slice_n,
            slice_start,
            n_slices,
            rank,
            n_tiles,
            max_n,
            win_off,
            win_len,
        )
    }
}

pub unsafe fn lora_shrink(
    stream: *mut c_void,
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
) -> i32 {
    unsafe {
        crate::nv_kernels_lora_shrink(
            stream,
            x_bf16,
            a_ptrs,
            buffer,
            token_lora_mapping,
            token_indices_sorted,
            num_tokens_per_lora,
            lora_token_start_loc,
            active_lora_ids,
            m,
            rank,
            k,
            n_slices,
            grid_loras,
            a_d0_stride,
            scale,
        )
    }
}

pub unsafe fn lora_fused(
    stream: *mut c_void,
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
) -> i32 {
    unsafe {
        crate::nv_kernels_lora_fused(
            stream,
            x_bf16,
            a_ptrs,
            b_ptrs,
            y_bf16,
            token_indices_sorted,
            num_tokens_per_lora,
            lora_token_start_loc,
            active_lora_ids,
            slice_n,
            slice_start,
            b_d0_stride,
            m,
            rank,
            k,
            max_n,
            n_slices,
            grid_loras,
            a_d0_stride,
            win_off,
            win_len,
            y_row_stride,
            scale,
        )
    }
}

pub unsafe fn lora_expand(
    stream: *mut c_void,
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
) -> i32 {
    unsafe {
        crate::nv_kernels_lora_expand(
            stream,
            buffer,
            b_ptrs,
            y_bf16,
            token_lora_mapping,
            token_indices_sorted,
            num_tokens_per_lora,
            lora_token_start_loc,
            active_lora_ids,
            slice_n,
            slice_start,
            m,
            rank,
            max_n,
            n_slices,
            grid_loras,
            y_row_stride,
        )
    }
}
