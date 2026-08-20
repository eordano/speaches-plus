#![cfg(feature = "cuda")]

use nv_kernels::cuda;
use std::ffi::c_void;
use std::ptr;

const NVK_ERR_GRID_AXIS: i32 = -9;
const OVER_LIMIT: i32 = 65536;

#[test]
fn moe_gemv_swiglu_bf16_m1_refuses_k_over_65535() {
    let rc = unsafe {
        cuda::moe_gemv_swiglu_bf16_m1(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            OVER_LIMIT,
            1,
            8,
            8,
        )
    };
    assert_eq!(
        rc, NVK_ERR_GRID_AXIS,
        "k={OVER_LIMIT} should be refused before the grid.y launch; got rc={rc}"
    );
}

#[test]
fn moe_gemv_swiglu_bf16_mb_refuses_k_over_65535() {
    let rc = unsafe {
        cuda::moe_gemv_swiglu_bf16_mb(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            1,
            OVER_LIMIT,
            1,
            8,
            8,
        )
    };
    assert_eq!(
        rc, NVK_ERR_GRID_AXIS,
        "k={OVER_LIMIT} should be refused before the grid.y launch; got rc={rc}"
    );
}

#[test]
fn moe_gemv_swiglu_bf16_mb_refuses_b_over_65535() {
    let rc = unsafe {
        cuda::moe_gemv_swiglu_bf16_mb(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            OVER_LIMIT,
            1,
            1,
            8,
            8,
        )
    };
    assert_eq!(
        rc, NVK_ERR_GRID_AXIS,
        "b={OVER_LIMIT} should be refused before the grid.z launch; got rc={rc}"
    );
}

#[test]
fn moe_gemv_down_tail_bf16_mb_refuses_b_over_65535() {
    let rc = unsafe {
        cuda::moe_gemv_down_tail_bf16_mb(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            OVER_LIMIT,
            1,
            1,
            8,
            8,
        )
    };
    assert_eq!(
        rc, NVK_ERR_GRID_AXIS,
        "b={OVER_LIMIT} should be refused before the grid.z launch; got rc={rc}"
    );
}

#[test]
fn moe_grouped_fp4_gemv_m1_bf16_refuses_num_groups_over_65535() {
    let rc = unsafe {
        cuda::moe_grouped_fp4_gemv_m1_bf16(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            ptr::null(),
            OVER_LIMIT,
            1,
            1,
            16,
            128,
            1,
        )
    };
    assert_eq!(
        rc, NVK_ERR_GRID_AXIS,
        "num_groups={OVER_LIMIT} should be refused before the grid.y launch; got rc={rc}"
    );
}
