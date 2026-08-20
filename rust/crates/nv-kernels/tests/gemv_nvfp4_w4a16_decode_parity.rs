#![cfg(feature = "cuda")]

mod common;
use common::stream;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;
use common::LcgMask23TwoSided as Lcg;
use common::assert_rows_close;
use common::host_w_f64;

fn host_dot_rows_f64(wf: &[f64], x: &[f64], n: usize, k: usize) -> Vec<f64> {
    (0..n)
        .map(|r| (0..k).map(|c| wf[r * k + c] * x[c]).sum())
        .collect()
}

#[test]
fn gemv_nvfp4_w4a16_dual_and_silu_match_the_swizzled_dequant_host_oracle() {
    let Some(stream) = stream("gemv_nvfp4_w4a16_decode") else {
        return;
    };
    let mut rng = Lcg(0x243f6a8885a308d3);
    let (n, k) = (256usize, 512usize);
    let kb = k / 16;
    let packed_a = rng.packed_nibbles(n * k / 2);
    let packed_b = rng.packed_nibbles(n * k / 2);
    let sc_lin_a = rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(n * kb);
    let sc_lin_b = rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(n * kb);
    let sc_sw_a = nv_quant::nvfp4::swizzle_scales(&sc_lin_a, n, kb);
    let sc_sw_b = nv_quant::nvfp4::swizzle_scales(&sc_lin_b, n, kb);
    let x_words = rng.bf16_words(k, 1.0);
    let (alpha_a, alpha_b) = (0.0125f32, 0.05f32);

    #[allow(deprecated)]
    let dw_a: CudaSlice<u8> = stream.clone_htod(&packed_a).unwrap();
    #[allow(deprecated)]
    let dw_b: CudaSlice<u8> = stream.clone_htod(&packed_b).unwrap();
    #[allow(deprecated)]
    let ds_a: CudaSlice<u8> = stream.clone_htod(&sc_sw_a).unwrap();
    #[allow(deprecated)]
    let ds_b: CudaSlice<u8> = stream.clone_htod(&sc_sw_b).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.clone_htod(&x_words).unwrap();
    let mut dy_a: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
    let mut dy_b: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();

    let rc = {
        let (pwa, _a) = dw_a.device_ptr(&stream);
        let (pwb, _b) = dw_b.device_ptr(&stream);
        let (psa, _c) = ds_a.device_ptr(&stream);
        let (psb, _d) = ds_b.device_ptr(&stream);
        let (px, _e) = dx.device_ptr(&stream);
        let (pya, _f) = dy_a.device_ptr_mut(&stream);
        let (pyb, _g) = dy_b.device_ptr_mut(&stream);
        unsafe {
            cuda::gemv_nvfp4_w4a16_dual_m1(
                stream.cu_stream() as *mut c_void,
                pwa as *const u8,
                psa as *const u8,
                pwb as *const u8,
                psb as *const u8,
                px as *const u16,
                pya as *mut u16,
                pyb as *mut u16,
                alpha_a,
                alpha_b,
                n as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gemv_nvfp4_w4a16_dual_m1 rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let ya = stream.memcpy_dtov(&dy_a).unwrap();
    #[allow(deprecated)]
    let yb = stream.memcpy_dtov(&dy_b).unwrap();

    let xf: Vec<f64> = x_words
        .iter()
        .map(|w| bf16::from_bits(*w).to_f32() as f64)
        .collect();
    let wa = host_w_f64(&packed_a, &sc_lin_a, n, k, alpha_a);
    let wb = host_w_f64(&packed_b, &sc_lin_b, n, k, alpha_b);
    assert_rows_close("dual gate arm", &ya, &host_dot_rows_f64(&wa, &xf, n, k), 2.0e-2);
    assert_rows_close("dual up arm", &yb, &host_dot_rows_f64(&wb, &xf, n, k), 2.0e-2);

    let gate_words = rng.bf16_words(k, 1.0);
    let up_words = rng.bf16_words(k, 1.0);
    #[allow(deprecated)]
    let dg: CudaSlice<u16> = stream.clone_htod(&gate_words).unwrap();
    #[allow(deprecated)]
    let du: CudaSlice<u16> = stream.clone_htod(&up_words).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
    let rc = {
        let (pw, _a) = dw_a.device_ptr(&stream);
        let (ps, _b) = ds_a.device_ptr(&stream);
        let (pg, _c) = dg.device_ptr(&stream);
        let (pu, _d) = du.device_ptr(&stream);
        let (py, _e) = dy.device_ptr_mut(&stream);
        unsafe {
            cuda::gemv_nvfp4_w4a16_silu_gate_up_in_m1(
                stream.cu_stream() as *mut c_void,
                pw as *const u8,
                ps as *const u8,
                pg as *const u16,
                pu as *const u16,
                py as *mut u16,
                alpha_a,
                n as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gemv_nvfp4_w4a16_silu_gate_up_in_m1 rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let y = stream.memcpy_dtov(&dy).unwrap();
    let act: Vec<f64> = gate_words
        .iter()
        .zip(up_words.iter())
        .map(|(g, u)| {
            let gf = bf16::from_bits(*g).to_f32();
            let uf = bf16::from_bits(*u).to_f32();
            let a = (gf / (1.0 + (-gf).exp())) * uf;
            bf16::from_f32(a).to_f32() as f64
        })
        .collect();
    assert_rows_close(
        "silu-in down arm",
        &y,
        &host_dot_rows_f64(&wa, &act, n, k),
        2.0e-2,
    );
}

#[test]
fn gemv_nvfp4_w4a16_refuses_ragged_k_and_mismatched_dual_pointers_without_launching() {
    let Some(stream) = stream("gemv_nvfp4_w4a16_refusal") else {
        return;
    };
    let rc = unsafe {
        cuda::gemv_nvfp4_w4a16_dual_m1(
            stream.cu_stream() as *mut c_void,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1.0,
            1.0,
            16,
            24,
        )
    };
    assert_eq!(rc, -1, "ragged K must refuse, got {rc}");
    let rc = unsafe {
        cuda::gemv_nvfp4_w4a16_silu_gate_up_in_m1(
            stream.cu_stream() as *mut c_void,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            1.0,
            16,
            30000 * 16,
        )
    };
    assert_eq!(rc, -1, "over-smem K must refuse, got {rc}");
}

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_KERNELS_BENCH=1 -- times the two-kernel nvfp4 W4A16 decode MLP on the q38 dense shapes (dual gate+up N=17408 K=5120, silu-in down N=5120 K=17408) so the per-layer cost is comparable against the padded-A4 tensor-core route the profiler measured at ~164 us/layer"]
fn gemv_nvfp4_w4a16_decode_mlp_shape_bench() {
    if std::env::var("NV_KERNELS_BENCH").as_deref() != Ok("1") {
        eprintln!("SKIP: set NV_KERNELS_BENCH=1 to run");
        return;
    }
    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    let (hidden, inter) = (5120usize, 17408usize);
    let packed_g = rng.packed_nibbles(inter * hidden / 2);
    let packed_u = rng.packed_nibbles(inter * hidden / 2);
    let packed_d = rng.packed_nibbles(hidden * inter / 2);
    let sc_g = rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(inter * hidden / 16);
    let sc_u = rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(inter * hidden / 16);
    let sc_d = rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(hidden * inter / 16);
    let sw_g = nv_quant::nvfp4::swizzle_scales(&sc_g, inter, hidden / 16);
    let sw_u = nv_quant::nvfp4::swizzle_scales(&sc_u, inter, hidden / 16);
    let sw_d = nv_quant::nvfp4::swizzle_scales(&sc_d, hidden, inter / 16);
    let x = rng.bf16_words(hidden, 1.0);
    #[allow(deprecated)]
    let dwg: CudaSlice<u8> = stream.clone_htod(&packed_g).unwrap();
    #[allow(deprecated)]
    let dwu: CudaSlice<u8> = stream.clone_htod(&packed_u).unwrap();
    #[allow(deprecated)]
    let dwd: CudaSlice<u8> = stream.clone_htod(&packed_d).unwrap();
    #[allow(deprecated)]
    let dsg: CudaSlice<u8> = stream.clone_htod(&sw_g).unwrap();
    #[allow(deprecated)]
    let dsu: CudaSlice<u8> = stream.clone_htod(&sw_u).unwrap();
    #[allow(deprecated)]
    let dsd: CudaSlice<u8> = stream.clone_htod(&sw_d).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
    let mut dg: CudaSlice<u16> = stream.alloc_zeros::<u16>(inter).unwrap();
    let mut du: CudaSlice<u16> = stream.alloc_zeros::<u16>(inter).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(hidden).unwrap();

    let one_layer = |stream: &Arc<CudaStream>,
                     dg: &mut CudaSlice<u16>,
                     du: &mut CudaSlice<u16>,
                     dy: &mut CudaSlice<u16>| {
        let (pwg, _a) = dwg.device_ptr(stream);
        let (pwu, _b) = dwu.device_ptr(stream);
        let (pwd, _c) = dwd.device_ptr(stream);
        let (psg, _d) = dsg.device_ptr(stream);
        let (psu, _e) = dsu.device_ptr(stream);
        let (psd, _f) = dsd.device_ptr(stream);
        let (px, _g) = dx.device_ptr(stream);
        let (pg, _h) = dg.device_ptr_mut(stream);
        let (pu, _i) = du.device_ptr_mut(stream);
        let (py, _j) = dy.device_ptr_mut(stream);
        let rc = unsafe {
            cuda::gemv_nvfp4_w4a16_dual_m1(
                stream.cu_stream() as *mut c_void,
                pwg as *const u8,
                psg as *const u8,
                pwu as *const u8,
                psu as *const u8,
                px as *const u16,
                pg as *mut u16,
                pu as *mut u16,
                0.01,
                0.01,
                inter as i32,
                hidden as i32,
            )
        };
        assert_eq!(rc, 0, "dual rc={rc}");
        let rc = unsafe {
            cuda::gemv_nvfp4_w4a16_silu_gate_up_in_m1(
                stream.cu_stream() as *mut c_void,
                pwd as *const u8,
                psd as *const u8,
                pg as *const u16,
                pu as *const u16,
                py as *mut u16,
                0.01,
                hidden as i32,
                inter as i32,
            )
        };
        assert_eq!(rc, 0, "silu rc={rc}");
    };

    for _ in 0..3 {
        one_layer(&stream, &mut dg, &mut du, &mut dy);
    }
    stream.synchronize().unwrap();
    let iters = 100usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        one_layer(&stream, &mut dg, &mut du, &mut dy);
    }
    stream.synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let weight_bytes = (3 * inter * hidden) / 2 + (3 * inter * hidden) / 16;
    let gbs = weight_bytes as f64 / us / 1e3;
    eprintln!(
        "NVFP4-W4A16-MLP-LAYER hidden={hidden} inter={inter} us_per_layer={us:.1} weight_gbs={gbs:.0}"
    );
}
