#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use float8::F8E4M3;
use half::bf16;
use nv_quant::fp8::{cpu_e4m3_matmul_weight_row, supports_fp8, Fp8GemmRunner};

fn detect_capability(ctx: &cudarc::driver::CudaContext) -> i32 {
    use cudarc::driver::sys::CUdevice_attribute;
    ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0)
}

#[test]
fn fp8_e4m3_matmul_matches_cpu_reference() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let major = detect_capability(&ctx);
    if !supports_fp8(major) {
        eprintln!("skip: SM {major} lacks FP8 support");
        return;
    }

    let (m, n, k) = (32usize, 64usize, 128usize);

    let mut a_host = Vec::with_capacity(m * k);
    let mut b_host = Vec::with_capacity(n * k);
    for i in 0..(m * k) {
        a_host.push(F8E4M3::from(((i as f32) * 0.013).sin()));
    }
    for i in 0..(n * k) {
        b_host.push(F8E4M3::from(((i as f32) * 0.017).cos()));
    }
    let a_scale_host = vec![0.5f32];
    let b_scale_host = vec![1.5f32];

    let a_bytes: Vec<u8> = a_host.iter().map(|x| x.to_bits()).collect();
    let b_bytes: Vec<u8> = b_host.iter().map(|x| x.to_bits()).collect();

    #[allow(deprecated)]
    let a_dev = stream.memcpy_stod(&a_bytes).unwrap();
    #[allow(deprecated)]
    let b_dev = stream.memcpy_stod(&b_bytes).unwrap();

    let mut d_dev = stream.alloc_zeros::<bf16>(m * n).unwrap();
    #[allow(deprecated)]
    let a_scale_dev = stream.memcpy_stod(&a_scale_host).unwrap();
    #[allow(deprecated)]
    let b_scale_dev = stream.memcpy_stod(&b_scale_host).unwrap();

    let mut runner = Fp8GemmRunner::new(stream.clone()).unwrap();
    runner
        .matmul_e4m3_weight_row(
            &a_dev,
            &b_dev,
            &mut d_dev,
            m as u64,
            n as u64,
            k as u64,
            &a_scale_dev,
            &b_scale_dev,
        )
        .unwrap();
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let got = stream.memcpy_dtov(&d_dev).unwrap();
    let expect =
        cpu_e4m3_matmul_weight_row(&a_host, &b_host, a_scale_host[0], b_scale_host[0], m, n, k);

    let mut max_abs = 0f32;
    for (g, e) in got.iter().zip(expect.iter()) {
        max_abs = max_abs.max((g.to_f32() - e.to_f32()).abs());
    }
    assert!(max_abs < 2.0, "fp8 matmul drift {max_abs} exceeds 2.0");
}
