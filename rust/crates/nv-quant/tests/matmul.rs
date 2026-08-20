#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use half::bf16;
use nv_quant::matmul::{cpu_bf16_matmul_row_major, TensorCoreGemm};

fn build_input(m: usize, n: usize, k: usize) -> (Vec<bf16>, Vec<bf16>) {
    let mut a = Vec::with_capacity(m * k);
    let mut b = Vec::with_capacity(k * n);
    for i in 0..(m * k) {
        a.push(bf16::from_f32((i as f32 * 0.0007).sin()));
    }
    for i in 0..(k * n) {
        b.push(bf16::from_f32((i as f32 * 0.0011).cos()));
    }
    (a, b)
}

#[test]
fn bf16_matmul_matches_cpu_reference() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let (m, n, k) = (64usize, 96usize, 128usize);
    let (a_host, b_host) = build_input(m, n, k);

    #[allow(deprecated)]
    let a_dev = stream.memcpy_stod(&a_host).unwrap();
    #[allow(deprecated)]
    let b_dev = stream.memcpy_stod(&b_host).unwrap();
    let mut c_dev = stream.alloc_zeros::<bf16>(m * n).unwrap();

    let gemm = TensorCoreGemm::new(stream.clone()).unwrap();
    gemm.bf16_matmul_row_major(
        &stream, &a_dev, &b_dev, &mut c_dev, m as u64, n as u64, k as u64, 1.0, 0.0,
    )
    .unwrap();
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let got = stream.memcpy_dtov(&c_dev).unwrap();
    let expect = cpu_bf16_matmul_row_major(&a_host, &b_host, m, n, k);

    let mut max_rel = 0f32;
    let mut max_abs = 0f32;
    for (g, e) in got.iter().zip(expect.iter()) {
        let g = g.to_f32();
        let e = e.to_f32();
        let abs = (g - e).abs();
        let rel = if e.abs() > 1e-6 { abs / e.abs() } else { abs };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_abs < 0.5 && max_rel < 0.05,
        "bf16 matmul drift max_abs={max_abs} max_rel={max_rel}"
    );
}
