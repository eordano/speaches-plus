#![cfg(feature = "cuda")]

mod common;
use candle_core::{DType, Device, Tensor};
use common::detect_major;
use cudarc::driver::CudaContext;
use half::bf16;
use nv_layers::linear::Linear;
use nv_quant::nvfp4::{supports_nvfp4, Nvfp4GemmRunner};
use std::sync::{Arc, Mutex};

const GEMMA4_DOWN_PROJ_N_5376: usize = 5376;
const GEMMA4_DOWN_PROJ_K_21504: usize = 21504;
const GEMMA4_QKV_FUSED_N_16384: usize = 16384;
const GEMMA4_HIDDEN_K_5376: usize = 5376;

const CHECKPOINT_L0_DOWN_PROJ_INPUT_SCALE_0_32589_STORED_AS_ITS_RECIPROCAL: f32 = 3.0685;

const COSINE_FLOOR_ACTIVATION_QUANT_NOISE_ALONE_MEASURED_0_9958_ON_REAL_WEIGHTS: f64 = 0.99;

const NORM_RATIO_BAND_A_MISSING_INPUT_GLOBAL_FOLD_LANDS_AT_0_33: (f64, f64) = (0.9, 1.1);

fn cosine_and_norm_ratio(reference: &[f32], candidate: &[f32]) -> (f64, f64) {
    let mut dot = 0f64;
    let mut nref = 0f64;
    let mut ncand = 0f64;
    for (a, b) in reference.iter().zip(candidate) {
        dot += (*a as f64) * (*b as f64);
        nref += (*a as f64).powi(2);
        ncand += (*b as f64).powi(2);
    }
    (
        dot / (nref.sqrt() * ncand.sqrt()).max(1e-30),
        (ncand / nref.max(1e-30)).sqrt(),
    )
}

fn host_f32(y: &Tensor) -> Vec<f32> {
    y.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn check_shape(
    device: &Device,
    runner: &Arc<Mutex<Nvfp4GemmRunner>>,
    n: usize,
    k: usize,
    label: &str,
) {
    let w: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32((i as f32 * 0.0009).cos() * 0.05))
        .collect();
    let w_t = Tensor::from_vec(w, (n, k), device).unwrap();
    let base = Linear::from_bf16_quantized_nvfp4_dev(&w_t, None, device, runner.clone()).unwrap();
    let (wq, sc, weight_alpha_over_unit_input_global, base_input_global) =
        base.nvfp4_parts().expect("base linear is NVFP4-resident");
    assert_eq!(
        base_input_global, 1.0,
        "from_bf16_quantized_nvfp4_dev stores a unit input global; the disk-semantics twin \
         below derives its scales from that assumption"
    );
    let g_x = CHECKPOINT_L0_DOWN_PROJ_INPUT_SCALE_0_32589_STORED_AS_ITS_RECIPROCAL;
    let disk_semantics_twin = Linear::new_nvfp4_scaled(
        wq.clone(),
        sc.clone(),
        k,
        n,
        None,
        device,
        runner.clone(),
        weight_alpha_over_unit_input_global / g_x,
        g_x,
    )
    .unwrap();

    let x: Vec<bf16> = (0..k)
        .map(|i| bf16::from_f32((i as f32 * 0.0007).sin()))
        .collect();
    let x_t = Tensor::from_vec(x, (1usize, k), device).unwrap();

    let y_mma = host_f32(&disk_semantics_twin.forward(&x_t).unwrap());
    let y_gemv = host_f32(
        &disk_semantics_twin
            .forward_nvfp4_gemv_m1_unconditionally_which_a_draft_only_lm_head_requires(&x_t)
            .unwrap(),
    );

    let mma_norm = y_mma.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
    assert!(
        mma_norm > 0.0,
        "{label}: the mma reference output is all zeros, so both scale and layout checks \
         below would pass vacuously"
    );
    let (cos, norm_ratio) = cosine_and_norm_ratio(&y_mma, &y_gemv);
    eprintln!("M1-GEMV-PARITY {label} n={n} k={k} cosine={cos:.6} norm_ratio={norm_ratio:.6}");
    assert!(
        cos >= COSINE_FLOOR_ACTIVATION_QUANT_NOISE_ALONE_MEASURED_0_9958_ON_REAL_WEIGHTS,
        "{label}: cosine {cos:.6} -- the two routes disagree in SHAPE, which is a weight \
         layout or scale-swizzle defect, not the global-scale defect this gate also covers"
    );
    let (lo, hi) = NORM_RATIO_BAND_A_MISSING_INPUT_GLOBAL_FOLD_LANDS_AT_0_33;
    assert!(
        norm_ratio >= lo && norm_ratio <= hi,
        "{label}: norm_ratio {norm_ratio:.6} outside [{lo}, {hi}] -- a constant factor of \
         ~{:.3} here means matmul_nvfp4_m1_gemv stopped folding input_stored_global into the \
         bf16-activation alpha (the mma route spends that global quantizing x; the gemv route \
         never quantizes x, so its alpha must repay it)",
        1.0 / g_x as f64
    );
}

#[test]
#[ignore = "GPU parity gate; set NV_NVFP4_M1_GEMV_PARITY=1 -- proves the m1 bf16-act gemv route matches the mma route on disk-checkpoint scale semantics (non-unit input global) at gemma4 serving shapes"]
fn m1_gemv_route_matches_mma_route_under_disk_checkpoint_input_globals_at_gemma4_shapes() {
    if std::env::var("NV_NVFP4_M1_GEMV_PARITY").as_deref() != Ok("1") {
        panic!(
            "NV_NVFP4_M1_GEMV_PARITY is not 1, so this parity gate executed nothing. It is \
             already #[ignore]d, which makes --ignored the opt-in; a second env gate behind that \
             can only make an --ignored run print `1 passed` for a gate that compared nothing. A \
             suite whose precondition is absent must fail or be ignored, never return green."
        );
    }
    assert!(
        std::env::var("NV_NVFP4_M1_GEMV").ok().as_deref() != Some("1"),
        "unset NV_NVFP4_M1_GEMV for this gate: with it set, Linear::forward at m=1 takes the \
         gemv route itself and the reference below compares the gemv route to itself"
    );
    let Ok(ctx) = CudaContext::new(0) else {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no CUDA context");
    };
    if !supports_nvfp4(detect_major(&ctx)) {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no NVFP4 support");
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream).unwrap()));

    check_shape(
        &device,
        &runner,
        GEMMA4_DOWN_PROJ_N_5376,
        GEMMA4_DOWN_PROJ_K_21504,
        "down_proj",
    );
    check_shape(
        &device,
        &runner,
        GEMMA4_QKV_FUSED_N_16384,
        GEMMA4_HIDDEN_K_5376,
        "qkv_fused",
    );
}
