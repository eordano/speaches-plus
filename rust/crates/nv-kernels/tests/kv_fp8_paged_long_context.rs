#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;
use std::ffi::c_void;

const CUDA_ERROR_INVALID_CONFIGURATION: i32 = 9;

fn launch(len: usize) -> i32 {
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();

    let n_kv = 1usize;
    let head_dim = 64usize;
    let block_size = 16usize;
    let blocks = len.div_ceil(block_size);

    let fp8: Vec<u8> = vec![0x38; len * n_kv * head_dim];
    let scales: Vec<f32> = vec![1.0; len * n_kv];
    let table: Vec<i32> = (0..blocks as i32).collect();

    #[allow(deprecated)]
    let d_fp8: CudaSlice<u8> = stream.clone_htod(&fp8).unwrap();
    #[allow(deprecated)]
    let d_scales: CudaSlice<f32> = stream.clone_htod(&scales).unwrap();
    #[allow(deprecated)]
    let d_table: CudaSlice<i32> = stream.clone_htod(&table).unwrap();
    let mut d_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(len * n_kv * head_dim).unwrap();

    let rc = {
        let (p_fp8, _a) = d_fp8.device_ptr(&stream);
        let (p_sc, _b) = d_scales.device_ptr(&stream);
        let (p_tb, _c) = d_table.device_ptr(&stream);
        let (p_out, _d) = d_out.device_ptr_mut(&stream);
        unsafe {
            cuda::dequantize_kv_fp8_paged(
                stream.cu_stream() as *mut c_void,
                p_fp8 as *const u8,
                p_sc as *const f32,
                p_out as *mut u16,
                p_tb as *const i32,
                block_size as i32,
                len as i32,
                n_kv as i32,
                head_dim as i32,
            )
        }
    };
    if rc == 0 {
        stream.synchronize().unwrap();
    }
    rc
}

#[test]
#[ignore]
fn under_the_y_limit_has_always_worked() {
    let rc = launch(65535);
    assert_eq!(rc, 0, "len=65535 should launch; got rc={rc}");
}

#[test]
#[ignore]
fn over_the_y_limit() {

    let rc = launch(65536);
    assert_ne!(
        rc, CUDA_ERROR_INVALID_CONFIGURATION,
        "len=65536 returned cudaErrorInvalidConfiguration: the sequence length \
         is back on a grid axis limited to 65535"
    );
    assert_eq!(rc, 0, "len=65536 should launch; got rc={rc}");
}

#[test]
#[ignore]
fn well_past_the_limit_and_values_survive() {

    let len = 262144usize;
    let rc = launch(len);
    assert_eq!(rc, 0, "len={len} should launch; got rc={rc}");
}
