#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;

const D_K_A: usize = 8192;
const D_K_B: usize = 6200;
const N_K: usize = 1;
const N_V: usize = 1;
const D_V: usize = 32;
const THREADS: usize = 8;
const ITERS: usize = 400;

struct Bufs {
    mixed: CudaSlice<u16>,
    z: CudaSlice<u16>,
    a: CudaSlice<u16>,
    b: CudaSlice<u16>,
    a_log: CudaSlice<u16>,
    dt_bias: CudaSlice<u16>,
    norm_w: CudaSlice<u16>,
    state: CudaSlice<f32>,
    out: CudaSlice<u16>,
}

fn alloc(stream: &Arc<cudarc::driver::CudaStream>, d_k: usize) -> Bufs {
    let key_dim = N_K * d_k;
    Bufs {
        mixed: stream.alloc_zeros::<u16>(2 * key_dim + N_V * D_V).unwrap(),
        z: stream.alloc_zeros::<u16>(N_V * D_V).unwrap(),
        a: stream.alloc_zeros::<u16>(N_V).unwrap(),
        b: stream.alloc_zeros::<u16>(N_V).unwrap(),
        a_log: stream.alloc_zeros::<u16>(N_V).unwrap(),
        dt_bias: stream.alloc_zeros::<u16>(N_V).unwrap(),
        norm_w: stream.alloc_zeros::<u16>(D_V).unwrap(),
        state: stream.alloc_zeros::<f32>(N_V * d_k * D_V).unwrap(),
        out: stream.alloc_zeros::<u16>(N_V * D_V).unwrap(),
    }
}

fn launch(stream: &Arc<cudarc::driver::CudaStream>, bufs: &mut Bufs, d_k: usize) -> i32 {
    let (mixed, _g0) = bufs.mixed.device_ptr(stream);
    let (z, _g1) = bufs.z.device_ptr(stream);
    let (a, _g2) = bufs.a.device_ptr(stream);
    let (b, _g3) = bufs.b.device_ptr(stream);
    let (a_log, _g4) = bufs.a_log.device_ptr(stream);
    let (dt_bias, _g5) = bufs.dt_bias.device_ptr(stream);
    let (norm_w, _g6) = bufs.norm_w.device_ptr(stream);
    let (state, _g7) = bufs.state.device_ptr_mut(stream);
    let (out, _g8) = bufs.out.device_ptr_mut(stream);
    unsafe {
        cuda::gdn_decode_step_bf16(
            stream.cu_stream() as *mut c_void,
            mixed as *const u16,
            z as *const u16,
            a as *const u16,
            b as *const u16,
            a_log as *const u16,
            dt_bias as *const u16,
            norm_w as *const u16,
            state as *mut f32,
            out as *mut u16,
            N_K as i32,
            N_V as i32,
            d_k as i32,
            D_V as i32,
            1e-6,
        )
    }
}

#[test]
fn concurrent_gdn_decode_shapes_never_lower_the_optin_under_each_other() {
    let ctx = CudaContext::new(0).expect("cuda device 0");

    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let ctx = ctx.clone();
            std::thread::spawn(move || {
                let stream = ctx.new_stream().expect("stream");
                let d_k = if t % 2 == 0 { D_K_A } else { D_K_B };
                let mut bufs = alloc(&stream, d_k);
                let mut bad = 0usize;
                let mut first_rc = 0i32;
                for _ in 0..ITERS {
                    let rc = launch(&stream, &mut bufs, d_k);
                    if rc != 0 {
                        if bad == 0 {
                            first_rc = rc;
                        }
                        bad += 1;
                    }
                }
                stream.synchronize().unwrap();
                (bad, first_rc)
            })
        })
        .collect();

    let mut total_bad = 0usize;
    let mut any_rc = 0i32;
    for h in handles {
        let (bad, rc) = h.join().expect("thread");
        total_bad += bad;
        if bad > 0 && any_rc == 0 {
            any_rc = rc;
        }
    }

    assert_eq!(
        total_bad,
        0,
        "{total_bad} of {} launches were rejected (first rc={any_rc}). Two shapes \
         sharing one kernel raced on cudaFuncAttributeMaxDynamicSharedMemorySize: \
         one thread lowered it between another's set and its launch. The opt-in \
         must only ever be raised.",
        THREADS * ITERS
    );
}
