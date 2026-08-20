#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice};
use half::bf16;
use nv_quant::nvfp4::Nvfp4GemmRunner;

#[path = "nvfp4_true_m_common.rs"]
mod common;

#[test]
fn default_config_lt_vs_cutlass_split_stays_within_measured_envelope() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    assert!(
        std::env::var("NV_NVFP4_SKINNY_LT").is_err()
            && std::env::var("NV_NVFP4_STREAMK").is_err(),
        "this test measures the SHIPPED default routing; unset NV_NVFP4_SKINNY_LT / NV_NVFP4_STREAMK"
    );
    let stream = ctx.default_stream();
    let mut runner = Nvfp4GemmRunner::new(stream.clone()).unwrap();

    let shapes = [(512usize, 5376usize), (128usize, 256usize)];
    let ms = [1usize, 4, 16, 64, 100, 127, 128];

    #[allow(deprecated)]
    let alpha_dev = stream.clone_htod(&[1.0f32]).unwrap();

    let mut worst = 0i64;
    let mut compared = 0usize;
    let mut ineligible: Vec<(usize, usize, usize)> = Vec::new();
    for (n, k) in shapes {
        let w_host: Vec<u16> = (0..n * k)
            .map(|j| bf16::from_f32(((j as f32) * 0.00071).cos()).to_bits())
            .collect();
        #[allow(deprecated)]
        let w_dev = stream.clone_htod(&w_host).unwrap();
        let (w_q, w_sf) = common::quantize_dev(&stream, &w_dev, n, n, k, 1.0);

        for &m in &ms {
            let a_host: Vec<u16> = (0..m * k)
                .map(|j| bf16::from_f32(((j as f32) * 0.00131).sin() * 3.0).to_bits())
                .collect();
            #[allow(deprecated)]
            let a_dev = stream.clone_htod(&a_host).unwrap();

            let m_pad = m.max(128);
            let (a_q_pad, a_sf_pad) = common::quantize_dev(&stream, &a_dev, m_pad, m, k, 1.0);
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

            if !runner.supports_true_m(m as u64, n as u64, k as u64) {
                eprintln!(
                    "[split] SKIPPED CELL m={m} n={n} k={k}: true-m ineligible -- this cell \
                     compared nothing"
                );
                ineligible.push((m, n, k));
                continue;
            }
            let (a_q, a_sf) = common::quantize_dev(&stream, &a_dev, m, m, k, 1.0);
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

            let hp: Vec<bf16> = stream.memcpy_dtov(&d_pad).unwrap();
            let ht: Vec<bf16> = stream.memcpy_dtov(&d_true).unwrap();
            let mut cell_worst = 0i64;
            let mut diffs = 0usize;
            for i in 0..m * n {
                let (p, t) = (hp[i].to_bits(), ht[i].to_bits());
                if p != t {
                    diffs += 1;
                    let d = (ord(p) - ord(t)).abs();
                    cell_worst = cell_worst.max(d);
                }
            }
            worst = worst.max(cell_worst);
            eprintln!(
                "[split] m={m:3} n={n:4} k={k:4}: {diffs} of {} elems differ, worst {cell_worst} bf16 ulp",
                m * n
            );

            assert!(
                cell_worst <= 4,
                "the shipped skinny-Lt-vs-CUTLASS routing split grew past its measured \
                 4-bf16-ulp envelope (m={m} n={n} k={k}: {cell_worst} ulp) -- see \
                 nvfp4_true_m's pin comment"
            );
            compared += 1;
        }
    }
    eprintln!(
        "[split] compared {compared} of {} cells (ineligible: {ineligible:?}); worst divergence \
         across the sweep: {worst} bf16 ulp",
        shapes.len() * ms.len()
    );
    assert!(
        compared > 0,
        "every one of the {} (m,n,k) cells took the `true-m ineligible` continue, so the shipped \
         routing split was never compared against anything and this test would have reported a \
         pass: {ineligible:?}",
        shapes.len() * ms.len()
    );
}

fn ord(bits: u16) -> i64 {
    let mag = (bits & 0x7fff) as i64;
    if bits & 0x8000 != 0 {
        -mag
    } else {
        mag
    }
}
