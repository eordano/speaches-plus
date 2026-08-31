#![cfg(feature = "cuda")]

use candle_core::Device;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use std::ffi::c_void;

const ITERS_200_ENOUGH_FOR_A_STABLE_MEDIAN_ON_A_QUIET_CARD: usize = 200;

fn bench_shape(dev: &candle_core::CudaDevice, n: usize, k: usize, label: &str) {
    let stream = nv_layers::cuda_stream::current_stream(dev);
    let packed_len = n * k / 2;
    let scales_len = n * (k / 16);
    let w: Vec<u8> = (0..packed_len).map(|i| (i % 251) as u8).collect();
    let s: Vec<u8> = (0..scales_len).map(|i| 0x30 + (i % 16) as u8).collect();
    let x: Vec<half::bf16> = (0..k)
        .map(|i| half::bf16::from_f32(((i % 13) as f32 - 6.0) * 0.01))
        .collect();
    let w_dev = stream.clone_htod(&w).expect("w htod");
    let s_dev = stream.clone_htod(&s).expect("s htod");
    let x_dev = stream.clone_htod(&x).expect("x htod");
    let mut y_dev = stream.alloc_zeros::<half::bf16>(n).expect("y alloc");

    let launch = |stream: &std::sync::Arc<cudarc::driver::CudaStream>,
                  w_dev: &cudarc::driver::CudaSlice<u8>,
                  s_dev: &cudarc::driver::CudaSlice<u8>,
                  x_dev: &cudarc::driver::CudaSlice<half::bf16>,
                  y_dev: &mut cudarc::driver::CudaSlice<half::bf16>| {
        let (wp, _g0) = w_dev.device_ptr(stream);
        let (sp, _g1) = s_dev.device_ptr(stream);
        let (xp, _g2) = x_dev.device_ptr(stream);
        let (yp, _g3) = y_dev.device_ptr_mut(stream);
        let rc = unsafe {
            nv_kernels::cuda::nvfp4_gemv_bf16act(
                stream.cu_stream() as *mut c_void,
                wp as *const u8,
                sp as *const u8,
                xp as *const u16,
                yp as *mut u16,
                1.0f32,
                n as i32,
                k as i32,
            )
        };
        assert_eq!(rc, 0, "nvfp4_gemv_bf16act rc={rc} for {n}x{k}");
    };

    for _ in 0..20 {
        launch(&stream, &w_dev, &s_dev, &x_dev, &mut y_dev);
    }
    stream.synchronize().expect("warm sync");
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS_200_ENOUGH_FOR_A_STABLE_MEDIAN_ON_A_QUIET_CARD {
        launch(&stream, &w_dev, &s_dev, &x_dev, &mut y_dev);
    }
    stream.synchronize().expect("timed sync");
    let us = t0.elapsed().as_secs_f64() * 1e6
        / ITERS_200_ENOUGH_FOR_A_STABLE_MEDIAN_ON_A_QUIET_CARD as f64;
    let bytes = packed_len + scales_len;
    let gbs = bytes as f64 / (us * 1e-6) / 1e9;
    eprintln!("G4-GEMV-FLOOR {label} n={n} k={k} us={us:.2} weight_gbs={gbs:.0}");
}

#[test]
#[ignore = "loads the 31B; set NV_G4_GEMV_DIAG=1 -- characterizes HOW the NV_NVFP4_M1_GEMV arm diverges from the default mma route on real checkpoint weights (constant factor => alpha bug, scrambled => layout bug)"]
fn gemma4_m1_gemv_vs_default_mma_route_divergence_shape_on_real_weights() {
    if std::env::var("NV_G4_GEMV_DIAG").as_deref() != Ok("1") {
        eprintln!("SKIP: set NV_G4_GEMV_DIAG=1");
        return;
    }
    use nv_models::gemma4::{Gemma4, Gemma4Config};
    use nv_weights::{QuantizationConfig, WeightLoader};
    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    let dir = std::fs::read_dir(&snaps)
        .expect("snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists())
        .expect("no 31B snapshot with config.json");
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Gemma4::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);

    let lin = &model.layers()[0].mlp.down_proj;
    let k = 21504usize;
    let x: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
    let x_t = candle_core::Tensor::from_vec(x, (1usize, k), &device)
        .expect("x")
        .to_dtype(candle_core::DType::BF16)
        .expect("bf16");
    let y_mma: Vec<f32> = lin
        .forward(&x_t)
        .expect("mma forward")
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flat")
        .to_vec1()
        .expect("host");
    let y_gemv: Vec<f32> = lin
        .forward_nvfp4_gemv_m1_unconditionally_which_a_draft_only_lm_head_requires(&x_t)
        .expect("gemv forward")
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flat")
        .to_vec1()
        .expect("host");
    let n = y_mma.len();
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut ratio_samples: Vec<f32> = Vec::new();
    for i in 0..n {
        dot += (y_mma[i] as f64) * (y_gemv[i] as f64);
        na += (y_mma[i] as f64).powi(2);
        nb += (y_gemv[i] as f64).powi(2);
        if y_mma[i].abs() > 1e-3 && ratio_samples.len() < 8 {
            ratio_samples.push(y_gemv[i] / y_mma[i]);
        }
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
    eprintln!(
        "G4-GEMV-DIAG down_proj n={n} cosine={cos:.6} norm_ratio={:.6} first_ratios={ratio_samples:?} y_mma[0..4]={:?} y_gemv[0..4]={:?}",
        (nb / na.max(1e-30)).sqrt(),
        &y_mma[0..4],
        &y_gemv[0..4]
    );
}

#[test]
#[ignore = "GPU micro-floor for the decode m1 nvfp4 gemv at the gemma4-31b projection shapes; set NV_G4_GEMV_FLOOR=1"]
fn gemma4_attnproj_m1_gemv_weight_bandwidth_at_serving_shapes() {
    if std::env::var("NV_G4_GEMV_FLOOR").as_deref() != Ok("1") {
        eprintln!("SKIP: set NV_G4_GEMV_FLOOR=1");
        return;
    }
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    bench_shape(&dev, 16384, 5376, "qkv_fused");
    bench_shape(&dev, 5376, 8192, "o_proj");
    bench_shape(&dev, 43008, 5376, "gate_up");
    bench_shape(&dev, 5376, 21504, "down");
    bench_shape(&dev, 262144, 5376, "lm_head_shape_ref");
}
