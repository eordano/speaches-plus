#![cfg(feature = "cuda")]

use nv_kernels::lora;
use std::ffi::c_void;
use std::ptr;

const NVK_ERR_GRID_AXIS: i32 = -9;

#[test]
fn lora_fused_n_slices_guard_is_refused() {
    let rc = unsafe {
        lora::lora_fused(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            1,
            1,
            1,
            1,
            65536,
            1,
            0,
            0,
            1,
            1,
            1.0,
        )
    };
    assert_eq!(rc, NVK_ERR_GRID_AXIS);
}

#[test]
fn lora_fused_grid_loras_guard_is_refused() {
    let rc = unsafe {
        lora::lora_fused(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            1,
            1,
            1,
            1,
            1,
            65536,
            0,
            0,
            1,
            1,
            1.0,
        )
    };
    assert_eq!(rc, NVK_ERR_GRID_AXIS);
}

#[test]
fn lora_shrink_n_slices_guard_is_refused() {
    let rc = unsafe {
        lora::lora_shrink(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            1,
            1,
            1,
            65536,
            1,
            0,
            1.0,
        )
    };
    assert_eq!(rc, NVK_ERR_GRID_AXIS);
}

#[test]
fn lora_shrink_grid_loras_guard_is_refused() {
    let rc = unsafe {
        lora::lora_shrink(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            1,
            1,
            1,
            1,
            65536,
            0,
            1.0,
        )
    };
    assert_eq!(rc, NVK_ERR_GRID_AXIS);
}

#[test]
fn lora_expand_n_slices_guard_is_refused() {
    let rc = unsafe {
        lora::lora_expand(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            1,
            1,
            1,
            65536,
            1,
            1,
        )
    };
    assert_eq!(rc, NVK_ERR_GRID_AXIS);
}

#[test]
fn lora_expand_grid_loras_guard_is_refused() {
    let rc = unsafe {
        lora::lora_expand(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            1,
            1,
            1,
            1,
            65536,
            1,
        )
    };
    assert_eq!(rc, NVK_ERR_GRID_AXIS);
}
