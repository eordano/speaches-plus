#![cfg(feature = "cuda")]

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::CudaContext;
use half::bf16;
use nv_quant::nvfp4::{cpu_nvfp4_matmul_weight_row, supports_nvfp4, Nvfp4GemmRunner, Nvfp4Tensor};

#[test]
fn nvfp4_block_scaled_matmul_matches_cpu_dequant() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major} lacks NVFP4 support");
        return;
    }
    let stream = ctx.default_stream();

    let (m, n, k) = (128usize, 128usize, 128usize);

    let a_rows: Vec<Vec<f32>> = (0..m)
        .map(|i| (0..k).map(|j| ((i * k + j) as f32 * 0.07).sin()).collect())
        .collect();
    let b_rows: Vec<Vec<f32>> = (0..n)
        .map(|i| (0..k).map(|j| ((i * k + j) as f32 * 0.09).cos()).collect())
        .collect();

    let a_q = Nvfp4Tensor::quantize_rows(&a_rows);
    let b_q = Nvfp4Tensor::quantize_rows(&b_rows);

    #[allow(deprecated)]
    let a_data = stream.memcpy_stod(&a_q.data).unwrap();
    #[allow(deprecated)]
    let a_scales = stream.memcpy_stod(&a_q.scales_swizzled()).unwrap();
    #[allow(deprecated)]
    let b_data = stream.memcpy_stod(&b_q.data).unwrap();
    #[allow(deprecated)]
    let b_scales = stream.memcpy_stod(&b_q.scales_swizzled()).unwrap();
    let mut d = stream.alloc_zeros::<bf16>(m * n).unwrap();

    let mut runner = Nvfp4GemmRunner::new(stream.clone()).unwrap();
    #[allow(deprecated)]
    let alpha_dev = stream.memcpy_stod(&[1.0f32]).unwrap();
    runner
        .matmul_scaled_alpha_dev(
            &a_data, &a_scales, &b_data, &b_scales, &mut d, m as u64, n as u64, k as u64,
            &alpha_dev, 1.0,
        )
        .unwrap();
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let got = stream.memcpy_dtov(&d).unwrap();
    let expect = cpu_nvfp4_matmul_weight_row(&a_q, &b_q, m, n, k);

    let mut sum_sq = 0f64;
    let mut sum_expect_sq = 0f64;
    for (g, e) in got.iter().zip(expect.iter()) {
        let d = (g.to_f32() - e.to_f32()) as f64;
        sum_sq += d * d;
        sum_expect_sq += (e.to_f32() as f64).powi(2);
    }
    let rel_rms = (sum_sq / sum_expect_sq.max(1e-12)).sqrt();
    assert!(
        rel_rms < 0.18,
        "nvfp4 relative-rms drift {rel_rms} exceeds 0.18"
    );
}

#[test]
fn nvfp4_block_scaled_matmul_matches_cpu_dequant_gaussian() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major} lacks NVFP4 support");
        return;
    }
    let stream = ctx.default_stream();

    let (m, n, k) = (128usize, 128usize, 128usize);

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((*state >> 11) as u32 & 0xFFFFFF) as f32 / (1u32 << 24) as f32;
        let v = ((state.wrapping_mul(2685821657736338717) >> 11) as u32 & 0xFFFFFF) as f32
            / (1u32 << 24) as f32;
        let u = u.max(1e-9);
        ((-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos())
    }
    let mut sa: u64 = 0xC0FFEE;
    let mut sb: u64 = 0xBADBED;
    let a_rows: Vec<Vec<f32>> = (0..m)
        .map(|_| (0..k).map(|_| lcg(&mut sa)).collect())
        .collect();
    let b_rows: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..k).map(|_| lcg(&mut sb)).collect())
        .collect();

    let a_q = Nvfp4Tensor::quantize_rows(&a_rows);
    let b_q = Nvfp4Tensor::quantize_rows(&b_rows);

    #[allow(deprecated)]
    let a_data = stream.memcpy_stod(&a_q.data).unwrap();
    #[allow(deprecated)]
    let a_scales = stream.memcpy_stod(&a_q.scales_swizzled()).unwrap();
    #[allow(deprecated)]
    let b_data = stream.memcpy_stod(&b_q.data).unwrap();
    #[allow(deprecated)]
    let b_scales = stream.memcpy_stod(&b_q.scales_swizzled()).unwrap();
    let mut d = stream.alloc_zeros::<bf16>(m * n).unwrap();

    let mut runner = Nvfp4GemmRunner::new(stream.clone()).unwrap();
    #[allow(deprecated)]
    let alpha_dev = stream.memcpy_stod(&[1.0f32]).unwrap();
    runner
        .matmul_scaled_alpha_dev(
            &a_data, &a_scales, &b_data, &b_scales, &mut d, m as u64, n as u64, k as u64,
            &alpha_dev, 1.0,
        )
        .unwrap();
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let got = stream.memcpy_dtov(&d).unwrap();
    let expect = cpu_nvfp4_matmul_weight_row(&a_q, &b_q, m, n, k);

    let mut sum_sq = 0f64;
    let mut sum_expect_sq = 0f64;
    for (g, e) in got.iter().zip(expect.iter()) {
        let d = (g.to_f32() - e.to_f32()) as f64;
        sum_sq += d * d;
        sum_expect_sq += (e.to_f32() as f64).powi(2);
    }
    let rel_rms = (sum_sq / sum_expect_sq.max(1e-12)).sqrt();
    eprintln!("nvfp4 gaussian rel_rms = {rel_rms}");
    assert!(
        rel_rms < 0.20,
        "nvfp4 relative-rms drift {rel_rms} exceeds 0.20 on gaussian data"
    );
}

#[test]
fn nvfp4_matmul_rejects_undersized_dims() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let mut runner = Nvfp4GemmRunner::new(stream.clone()).unwrap();
    let dummy_a = stream.alloc_zeros::<u8>(64).unwrap();
    let dummy_b = stream.alloc_zeros::<u8>(64).unwrap();
    let dummy_sa = stream.alloc_zeros::<u8>(64).unwrap();
    let dummy_sb = stream.alloc_zeros::<u8>(64).unwrap();
    let mut dummy_d = stream.alloc_zeros::<bf16>(64).unwrap();
    #[allow(deprecated)]
    let alpha_dev = stream.memcpy_stod(&[1.0f32]).unwrap();
    let err = runner
        .matmul_scaled_alpha_dev(
            &dummy_a,
            &dummy_sa,
            &dummy_b,
            &dummy_sb,
            &mut dummy_d,
            64,
            64,
            64,
            &alpha_dev,
            1.0,
        )
        .unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("128"), "unexpected error: {msg}");
}

#[test]
fn nvfp4_quant_dequant_round_trip_preserves_signal() {
    let row: Vec<f32> = (0..64).map(|i| (i as f32 * 0.12).sin()).collect();
    let q = Nvfp4Tensor::quantize_rows(&[row.clone()]);
    let deq = q.dequantize();
    let mut max_rel = 0f32;
    for (g, e) in deq[0].iter().zip(row.iter()) {
        let denom = e.abs().max(0.1);
        max_rel = max_rel.max((g - e).abs() / denom);
    }
    assert!(max_rel < 0.35, "nvfp4 quant rel err {max_rel}");
}
