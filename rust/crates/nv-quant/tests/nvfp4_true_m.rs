#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice};
use half::bf16;
use nv_quant::nvfp4::Nvfp4GemmRunner;

#[path = "nvfp4_true_m_common.rs"]
mod common;
use common::quantize_dev;

#[test]
fn nvfp4_true_m_matches_padded() {
    std::env::set_var("NV_NVFP4_SKINNY_LT", "0");
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let mut runner = Nvfp4GemmRunner::new(stream.clone()).unwrap();

    let shapes = [(512usize, 5376usize), (128usize, 256usize)];

    let ms = [1usize, 2, 3, 4, 5, 8, 9, 13, 16, 33, 64, 100, 127, 128];

    #[allow(deprecated)]
    let alpha_dev = stream.clone_htod(&[1.0f32]).unwrap();

    for (n, k) in shapes {
        let w_host: Vec<u16> = (0..n * k)
            .map(|j| bf16::from_f32(((j as f32) * 0.00071).cos()).to_bits())
            .collect();
        #[allow(deprecated)]
        let w_dev = stream.clone_htod(&w_host).unwrap();
        let (w_q, w_sf) = quantize_dev(&stream, &w_dev, n, n, k, 1.0);

        for &m in &ms {
            assert!(
                runner.supports_true_m(m as u64, n as u64, k as u64),
                "shape (m={m}, n={n}, k={k}) should be true-m eligible"
            );
            let a_host: Vec<u16> = (0..m * k)
                .map(|j| bf16::from_f32(((j as f32) * 0.00131).sin() * 3.0).to_bits())
                .collect();
            #[allow(deprecated)]
            let a_dev = stream.clone_htod(&a_host).unwrap();

            let m_pad = m.max(128);
            let (a_q_pad, a_sf_pad) = quantize_dev(&stream, &a_dev, m_pad, m, k, 1.0);
            let mut d_pad: CudaSlice<bf16> = stream.alloc_zeros::<bf16>(m_pad * n).unwrap();
            runner
                .matmul_scaled_alpha_dev(
                    &a_q_pad,
                    &a_sf_pad,
                    &w_q,
                    &w_sf,
                    &mut d_pad,
                    m_pad as u64,
                    n as u64,
                    k as u64,
                    &alpha_dev,
                    1.0,
                )
                .unwrap();

            let (a_q, a_sf) = quantize_dev(&stream, &a_dev, m, m, k, 1.0);
            let mut d_true: CudaSlice<bf16> = stream.alloc_zeros::<bf16>(m * n).unwrap();
            runner
                .matmul_scaled_alpha_dev(
                    &a_q,
                    &a_sf,
                    &w_q,
                    &w_sf,
                    &mut d_true,
                    m as u64,
                    n as u64,
                    k as u64,
                    &alpha_dev,
                    1.0,
                )
                .unwrap();

            stream.synchronize().unwrap();
            #[allow(deprecated)]
            let h_pad = stream.memcpy_dtov(&d_pad).unwrap();
            #[allow(deprecated)]
            let h_true = stream.memcpy_dtov(&d_true).unwrap();

            let mut mismatches = 0usize;
            let mut first = String::new();
            for row in 0..m {
                for col in 0..n {
                    let p = h_pad[row * n + col];
                    let t = h_true[row * n + col];
                    if p.to_bits() != t.to_bits() {
                        if mismatches == 0 {
                            first = format!(
                                "n={n} k={k} m={m} row={row} col={col}: padded={} true_m={}",
                                p.to_f32(),
                                t.to_f32()
                            );
                        }
                        mismatches += 1;
                    }
                }
            }
            assert_eq!(mismatches, 0, "true-m vs padded bitwise mismatch: {first}");
        }
        eprintln!("nvfp4 true-m n={n} k={k}: bitwise identical across m sweep");
    }
}
