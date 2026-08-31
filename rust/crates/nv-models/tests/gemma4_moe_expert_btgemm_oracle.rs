#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_layers::linear::Linear;

mod ctx_timing_common;

const GATE_UP_SHAPE_OUT_1408_IN_2816_THE_REAL_G4MOE_EXPERT: (usize, usize) = (1408, 2816);
const DOWN_SHAPE_OUT_2816_IN_704_THE_REAL_G4MOE_EXPERT: (usize, usize) = (2816, 704);
const M_LADDER_COVERS_GEMV_1_DET_2_16_AND_CUBLASLT_17_UP: &[usize] = &[1, 2, 8, 16, 17, 64];
const M_256_CROSS_ROUTE_ONLY_BECAUSE_THE_F64_HOST_ORACLE_IS_QUADRATIC: usize = 256;
const BAND_ABS_1E2_PLUS_REL_2PCT_MATCHING_FLASH_FP8_KVSHARE_PARITY: (f32, f32) = (1e-2, 0.02);

const BENCH_EXPERTS_128_LAYERS_30_THE_REAL_G4MOE_STACK: (usize, usize) = (128, 30);

fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

fn unit_vec(state: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| ((lcg(state) >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0)
        .collect()
}

fn device_or_refuse() -> Option<Device> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d),
        Err(e) => {
            if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() == Ok("1") {
                eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): no CUDA device 0: {e}");
                return None;
            }
            panic!(
                "gemma4_moe_expert_btgemm_oracle: no CUDA device 0: {e}. This is a correctness \
                 gate; it refuses to report success without running. Set NV_MODELS_ALLOW_SKIP=1 \
                 to skip on purpose."
            );
        }
    }
}

fn bf16_rounded_host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn host_oracle_x_wt_f64(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += x[i * k + p] as f64 * w[j * k + p] as f64;
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

fn check_shape(device: &Device, out_f: usize, in_f: usize, m: usize, with_oracle: bool) {
    let mut st = 0xb76e_0000_u64 ^ ((out_f as u64) << 32) ^ ((in_f as u64) << 16) ^ m as u64;
    let w_dev = Tensor::from_vec(unit_vec(&mut st, out_f * in_f), (out_f, in_f), device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let x_dev = Tensor::from_vec(unit_vec(&mut st, m * in_f), (m, in_f), device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let old_route = Linear::new(w_dev.clone(), None).unwrap();
    let bt_route = Linear::new_no_pretranspose(w_dev.clone(), None).unwrap();
    let y_old = bf16_rounded_host(&old_route.forward(&x_dev).unwrap());
    let y_bt = bf16_rounded_host(&bt_route.forward(&x_dev).unwrap());

    let (abs_band, rel_band) = BAND_ABS_1E2_PLUS_REL_2PCT_MATCHING_FLASH_FP8_KVSHARE_PARITY;
    if with_oracle {
        let w_host = bf16_rounded_host(&w_dev);
        let x_host = bf16_rounded_host(&x_dev);
        let reference = host_oracle_x_wt_f64(&x_host, &w_host, m, out_f, in_f);
        let max_abs_ref = reference.iter().fold(0f32, |a, v| a.max(v.abs()));
        for (label, y) in [("pretransposed", &y_old), ("bt", &y_bt)] {
            let mut worst = 0f32;
            for (g, r) in y.iter().zip(reference.iter()) {
                assert!(
                    g.is_finite(),
                    "out={out_f} in={in_f} m={m} route={label}: non-finite output"
                );
                let d = (g - r).abs();
                worst = worst.max(d);
                assert!(
                    d <= abs_band + rel_band * max_abs_ref,
                    "out={out_f} in={in_f} m={m} route={label}: |{g} - {r}| = {d} beyond \
                     abs {abs_band} + rel {rel_band} of max|ref| {max_abs_ref}; the band is \
                     row-scaled because bf16 accumulation error scales with the summands, not \
                     the cancelled result -- a route/indexing bug still shows as O(max|ref|)"
                );
            }
            println!(
                "BTGEMM-ORACLE out={out_f} in={in_f} m={m} route={label} worst_abs_diff={worst:.4} \
                 max_abs_ref={max_abs_ref:.3}"
            );
        }
    }
    let max_mag = y_old.iter().fold(0f32, |a, v| a.max(v.abs()));
    let cross = y_old
        .iter()
        .zip(y_bt.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        cross <= abs_band + rel_band * max_mag,
        "out={out_f} in={in_f} m={m}: pretransposed vs bt route max diff {cross} \
         (max|old|={max_mag}); both consume the same bf16 weight so beyond \
         reduction-order rounding means the bt plumbing reads the wrong layout"
    );
    println!("BTGEMM-CROSS out={out_f} in={in_f} m={m} cross_max_diff={cross:.4} max_mag={max_mag:.3}");
}

#[test]
fn expert_shapes_bt_route_matches_pretransposed_and_the_f64_oracle_across_the_m_ladder() {
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let Some(device) = device_or_refuse() else { return };
    for &(out_f, in_f) in &[
        GATE_UP_SHAPE_OUT_1408_IN_2816_THE_REAL_G4MOE_EXPERT,
        DOWN_SHAPE_OUT_2816_IN_704_THE_REAL_G4MOE_EXPERT,
    ] {
        for &m in M_LADDER_COVERS_GEMV_1_DET_2_16_AND_CUBLASLT_17_UP {
            check_shape(&device, out_f, in_f, m, true);
        }
        check_shape(
            &device,
            out_f,
            in_f,
            M_256_CROSS_ROUTE_ONLY_BECAUSE_THE_F64_HOST_ORACLE_IS_QUADRATIC,
            false,
        );
    }
}

#[test]
#[ignore = "isolated transpose-traffic bench at the real stacked shapes (needs a free 96GB-class \
            card); set NV_G4MOE_PREFILL_TEST=1 -- times one chunk-equivalent of the default \
            path's per-expert weight.t().contiguous() (all 128 experts x 30 layers) so the \
            wall share of the named ~45 GB/chunk waste is a measurement, not an estimate"]
fn transpose_traffic_one_chunk_equivalent_at_real_stacked_shapes() {
    if std::env::var("NV_G4MOE_PREFILL_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_G4MOE_PREFILL_TEST=1, \
                 or NV_MODELS_ALLOW_SKIP=1 to skip on purpose"
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_G4MOE_PREFILL_TEST=1 to run");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let Some(device) = device_or_refuse() else { return };
    let (experts, layers) = BENCH_EXPERTS_128_LAYERS_30_THE_REAL_G4MOE_STACK;
    let (gu_out, gu_in) = GATE_UP_SHAPE_OUT_1408_IN_2816_THE_REAL_G4MOE_EXPERT;
    let (dn_out, dn_in) = DOWN_SHAPE_OUT_2816_IN_704_THE_REAL_G4MOE_EXPERT;
    let gate_up = Tensor::zeros((experts, gu_out, gu_in), DType::BF16, &device).unwrap();
    let down = Tensor::zeros((experts, dn_out, dn_in), DType::BF16, &device).unwrap();

    let one_pass = |timed: bool| -> f64 {
        let t0 = std::time::Instant::now();
        for _layer in 0..layers {
            for e in 0..experts {
                let _wt = gate_up
                    .narrow(0, e, 1)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .t()
                    .unwrap()
                    .contiguous()
                    .unwrap();
                let _wt2 = down
                    .narrow(0, e, 1)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .t()
                    .unwrap()
                    .contiguous()
                    .unwrap();
            }
        }
        device.synchronize().unwrap();
        if timed {
            t0.elapsed().as_secs_f64()
        } else {
            0.0
        }
    };
    one_pass(false);
    let s = one_pass(true);
    let bytes = (layers * experts * (gu_out * gu_in + dn_out * dn_in) * 2) as f64;
    println!(
        "BTGEMM-TRANSPOSE-BENCH experts={experts} layers={layers} copied_gb={:.2} \
         chunk_equivalent_s={s:.3} effective_gbps={:.0}",
        bytes / 1e9,
        bytes / 1e9 / s,
    );
}
