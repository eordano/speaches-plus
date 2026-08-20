#![cfg(feature = "cuda")]

mod common;
use common::assert_u16_bits;
use common::dtoh_u16;
use common::htod_f32;
use common::htod_u16;
use common::lcg_unit_f32 as lcg;
use common::rand_bf16;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;

const TOKENS_3_EXERCISES_THE_STATE_SHIFT_TWICE: usize = 3;

fn rand_e4m3_bytes_never_nan(seed: &mut u64, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let mut b = (lcg(seed) * 255.0) as u8;
            if (b & 0x7f) == 0x7f {
                b ^= 0x01;
            }
            b
        })
        .collect()
}

fn rand_scales(seed: &mut u64, n: usize) -> Vec<f32> {
    (0..n).map(|_| 0.001 + lcg(seed) * 0.02).collect()
}

fn htod_u8(stream: &Arc<CudaStream>, v: &[u8]) -> CudaSlice<u8> {
    #[allow(deprecated)]
    let d = stream.clone_htod(&v.to_vec()).unwrap();
    d
}

fn run_case(name: &str, conv_dim: usize, value_dim: usize, k: usize, k_c: usize) {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("[{name}] skip: no CUDA device");
        return;
    };
    let stream = ctx.default_stream();
    let raw = stream.cu_stream() as *mut c_void;
    let n = conv_dim + value_dim;
    let mut seed = 0x9e3779b97f4a7c15u64 ^ (n as u64);

    let wq_host = rand_e4m3_bytes_never_nan(&mut seed, n * k);
    let scales_host = rand_scales(&mut seed, n);
    let conv_w_host = rand_bf16(&mut seed, conv_dim * k_c, -0.5, 0.5);
    let conv_state_host = rand_bf16(&mut seed, conv_dim * (k_c - 1), -1.0, 1.0);

    let wq = htod_u8(&stream, &wq_host);
    let scales = htod_f32(&stream, &scales_host);
    let conv_w = htod_u16(&stream, &conv_w_host);
    let mut state_ref = htod_u16(&stream, &conv_state_host);
    let mut state_fused = htod_u16(&stream, &conv_state_host);

    let mut y_ref: CudaSlice<u16> = stream.alloc_zeros(n).unwrap();
    let mut mixed_ref: CudaSlice<u16> = stream.alloc_zeros(conv_dim).unwrap();
    let mut mixed_fused: CudaSlice<u16> = stream.alloc_zeros(conv_dim).unwrap();
    let mut z_fused: CudaSlice<u16> = stream.alloc_zeros(value_dim).unwrap();

    for tok in 0..TOKENS_3_EXERCISES_THE_STATE_SHIFT_TWICE {
        let x_host = rand_bf16(&mut seed, k, -1.0, 1.0);
        let x = htod_u16(&stream, &x_host);

        let rc = unsafe {
            let (wp, _g1) = wq.device_ptr(&stream);
            let (sp, _g2) = scales.device_ptr(&stream);
            let (xp, _g3) = x.device_ptr(&stream);
            let (yp, _g4) = y_ref.device_ptr_mut(&stream);
            cuda::gemv_e4m3_mk(
                raw,
                wp as *const u8,
                sp as *const f32,
                xp as *const u16,
                yp as *mut u16,
                n as i32,
                k as i32,
                1,
            )
        };
        assert_eq!(rc, 0, "[{name}] tok {tok}: gemv_e4m3_mk rc={rc}");
        let rc = unsafe {
            let (yp, _g1) = y_ref.device_ptr(&stream);
            let (cs, _g2) = state_ref.device_ptr_mut(&stream);
            let (cw, _g3) = conv_w.device_ptr(&stream);
            let (mp, _g4) = mixed_ref.device_ptr_mut(&stream);
            cuda::gdn_conv_decode_silu_bf16(
                raw,
                yp as *const u16,
                cs as *mut u16,
                cw as *const u16,
                mp as *mut u16,
                conv_dim as i32,
                k_c as i32,
            )
        };
        assert_eq!(rc, 0, "[{name}] tok {tok}: gdn_conv_decode_silu_bf16 rc={rc}");

        let rc = unsafe {
            let (wp, _g1) = wq.device_ptr(&stream);
            let (sp, _g2) = scales.device_ptr(&stream);
            let (xp, _g3) = x.device_ptr(&stream);
            let (cw, _g4) = conv_w.device_ptr(&stream);
            let (cs, _g5) = state_fused.device_ptr_mut(&stream);
            let (mp, _g6) = mixed_fused.device_ptr_mut(&stream);
            let (zp, _g7) = z_fused.device_ptr_mut(&stream);
            cuda::gemv_e4m3_qkvz_conv_m1(
                raw,
                wp as *const u8,
                sp as *const f32,
                xp as *const u16,
                cw as *const u16,
                cs as *mut u16,
                mp as *mut u16,
                zp as *mut u16,
                n as i32,
                k as i32,
                conv_dim as i32,
                k_c as i32,
            )
        };
        assert_eq!(rc, 0, "[{name}] tok {tok}: gemv_e4m3_qkvz_conv_m1 rc={rc}");
        stream.synchronize().unwrap();

        let y_host = dtoh_u16(&stream, &y_ref);
        assert_u16_bits(
            &format!("[{name}] tok {tok} mixed"),
            &dtoh_u16(&stream, &mixed_fused),
            &dtoh_u16(&stream, &mixed_ref),
        );
        assert_u16_bits(
            &format!("[{name}] tok {tok} z"),
            &dtoh_u16(&stream, &z_fused),
            &y_host[conv_dim..],
        );
        assert_u16_bits(
            &format!("[{name}] tok {tok} conv state"),
            &dtoh_u16(&stream, &state_fused),
            &dtoh_u16(&stream, &state_ref),
        );
    }
}

#[test]
fn qkvz_conv_fused_gemv_is_bitwise_equal_to_gemv_then_conv() {
    run_case("qwen38ish", 2560, 1536, 512, 4);
}

#[test]
fn qkvz_conv_fused_gemv_ragged_rows_and_k2() {
    run_case("ragged", 300, 100, 256, 2);
}
