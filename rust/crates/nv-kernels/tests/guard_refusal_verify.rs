#![cfg(feature = "cuda")]

use nv_kernels::cuda;
use std::ffi::c_void;
use std::ptr;

const NVK_ERR_GRID_AXIS: i32 = -9;
const OVER_LIMIT_K: i32 = 65536;

#[test]
fn tree_verify_attn_bf16_refuses_k_over_65535() {
    let rc = unsafe {
        cuda::tree_verify_attn_bf16(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            1,
            1,
            1,
            OVER_LIMIT_K,
            0,
        )
    };
    assert_eq!(
        rc, NVK_ERR_GRID_AXIS,
        "K={OVER_LIMIT_K} should be refused before the grid.y launch; got rc={rc}"
    );
}

#[test]
fn tree_verify_attn_fp8_refuses_k_over_65535() {
    let rc = unsafe {
        cuda::tree_verify_attn_fp8(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            1,
            1,
            1,
            OVER_LIMIT_K,
            0,
            0,
        )
    };
    assert_eq!(
        rc, NVK_ERR_GRID_AXIS,
        "K={OVER_LIMIT_K} should be refused before the grid.y launch; got rc={rc}"
    );
}

#[test]
fn kv_append_fp8_refuses_k_over_65535() {
    let rc = unsafe {
        cuda::kv_append_fp8(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null(),
            OVER_LIMIT_K,
            1,
            1,
            0,
        )
    };
    assert_eq!(
        rc, NVK_ERR_GRID_AXIS,
        "K={OVER_LIMIT_K} should be refused before the grid.y launch; got rc={rc}"
    );
}

#[test]
fn verify_qkv_prep_refuses_k_over_65535() {
    let rc = unsafe {
        cuda::verify_qkv_prep(
            ptr::null_mut::<c_void>(),
            ptr::null(),
            0,
            0,
            0,
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            1e-6,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null(),
            OVER_LIMIT_K,
            1,
            1,
            2,
            0,
        )
    };
    assert_eq!(
        rc, NVK_ERR_GRID_AXIS,
        "K={OVER_LIMIT_K} should be refused before the grid.y launch; got rc={rc}"
    );
}
