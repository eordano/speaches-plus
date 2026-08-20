#![cfg(feature = "cuda")]

mod common;
use common::assert_u16_bits;
use common::dtoh_u16;
use common::htod_u16;
use common::rand_bf16;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;

fn run_case(name: &str, batch: usize, hidden: usize) {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("[{name}] skip: no CUDA device");
        return;
    };
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as *mut c_void;
    let n = batch * hidden;
    let mut seed = 0x1234_5678_9abc_def0u64 ^ (n as u64);

    let x_host = rand_bf16(&mut seed, n, -2.0, 2.0);
    let res_host = rand_bf16(&mut seed, n, -2.0, 2.0);
    let w_host = rand_bf16(&mut seed, hidden, 0.5, 1.5);

    let x = htod_u16(&stream, &x_host);
    let w = htod_u16(&stream, &w_host);

    let mut res_inplace = htod_u16(&stream, &res_host);
    let mut out_inplace: CudaSlice<u16> = stream.alloc_zeros(n).unwrap();
    let rc = unsafe {
        let (px, _g1) = x.device_ptr(&stream);
        let (pr, _g2) = res_inplace.device_ptr_mut(&stream);
        let (pw, _g3) = w.device_ptr(&stream);
        let (po, _g4) = out_inplace.device_ptr_mut(&stream);
        cuda::rmsnorm_residual_bf16(
            raw,
            px as *const u16,
            pr as *mut u16,
            pw as *const u16,
            po as *mut u16,
            batch,
            hidden,
            1e-6,
        )
    };
    assert_eq!(rc, 0, "[{name}] rmsnorm_residual_bf16 rc={rc}");

    let res_in = htod_u16(&stream, &res_host);
    let mut res_out: CudaSlice<u16> = stream.alloc_zeros(n).unwrap();
    let mut out_writeout: CudaSlice<u16> = stream.alloc_zeros(n).unwrap();
    let rc = unsafe {
        let (px, _g1) = x.device_ptr(&stream);
        let (pri, _g2) = res_in.device_ptr(&stream);
        let (pw, _g3) = w.device_ptr(&stream);
        let (pro, _g4) = res_out.device_ptr_mut(&stream);
        let (po, _g5) = out_writeout.device_ptr_mut(&stream);
        cuda::rmsnorm_residual_writeout_bf16(
            raw,
            px as *const u16,
            pri as *const u16,
            pw as *const u16,
            pro as *mut u16,
            po as *mut u16,
            batch,
            hidden,
            1e-6,
        )
    };
    assert_eq!(rc, 0, "[{name}] rmsnorm_residual_writeout_bf16 rc={rc}");
    stream.synchronize().unwrap();

    let out_a = dtoh_u16(&stream, &out_inplace);
    let out_b = dtoh_u16(&stream, &out_writeout);
    let res_a = dtoh_u16(&stream, &res_inplace);
    let res_b = dtoh_u16(&stream, &res_out);
    let live = out_a.iter().filter(|v| **v != 0).count();
    assert!(
        live > out_a.len() / 2,
        "[{name}] inplace reference output is mostly zero ({live}/{}); dead reference",
        out_a.len()
    );
    assert_u16_bits(&format!("{name}: normed"), &out_b, &out_a);
    assert_u16_bits(&format!("{name}: residual"), &res_b, &res_a);
}

#[test]
fn rmsnorm_residual_writeout_bitwise_m1_hidden_2048() {
    run_case("m1_h2048", 1, 2048);
}

#[test]
fn rmsnorm_residual_writeout_bitwise_rows_5_hidden_512() {
    run_case("r5_h512", 5, 512);
}

#[test]
fn rmsnorm_residual_writeout_bitwise_hidden_not_multiple_of_block() {
    run_case("r3_h300", 3, 300);
}
