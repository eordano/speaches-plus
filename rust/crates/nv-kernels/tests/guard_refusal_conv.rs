#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::ptr;

const NVK_ERR_GRID_AXIS: i32 = -9;

#[test]
fn depthwise_conv1d_c_guard_is_refused() {
    let rc = unsafe {
        nv_kernels::cuda::depthwise_conv1d_silu_bf16(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            1,
            65536,
            1,
            3,
        )
    };
    assert_eq!(rc, NVK_ERR_GRID_AXIS);
}

#[test]
fn depthwise_conv1d_tile_count_guard_is_refused() {
    let rc = unsafe {
        nv_kernels::cuda::depthwise_conv1d_silu_bf16(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            1,
            1,
            65535 * 256 + 1,
            3,
        )
    };
    assert_eq!(rc, NVK_ERR_GRID_AXIS);
}
