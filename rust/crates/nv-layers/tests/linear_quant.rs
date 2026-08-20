#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::CudaContext;
use nv_layers::linear::Linear;
use nv_quant::fp8::{supports_fp8, Fp8GemmRunner};
use nv_quant::nvfp4::{supports_nvfp4, Nvfp4GemmRunner};
use std::sync::{Arc, Mutex};

fn detect_major(ctx: &cudarc::driver::CudaContext) -> i32 {
    ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0)
}

const GEMMA4_31B_NVFP4_REPO: &str = "models--nvidia--Gemma-4-31B-IT-NVFP4";

fn hub_roots() -> Vec<std::path::PathBuf> {
    let mut roots = Vec::new();
    if let Some(h) = std::env::var_os("HF_HUB_CACHE") {
        roots.push(std::path::PathBuf::from(h));
    }
    if let Some(h) = std::env::var_os("HOME") {
        roots.push(std::path::PathBuf::from(h).join(".cache/huggingface/hub"));
    }
    roots
}

fn hub_snapshot_with(repo: &str, marker: &str) -> Option<std::path::PathBuf> {
    for root in hub_roots() {
        let snaps = root.join(repo).join("snapshots");
        if let Ok(sha) = std::fs::read_to_string(root.join(repo).join("refs/main")) {
            let p = snaps.join(sha.trim());
            if p.join(marker).exists() {
                return Some(p);
            }
        }
        let mut cands: Vec<std::path::PathBuf> = std::fs::read_dir(&snaps)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.join(marker).exists())
            .collect();
        cands.sort();
        if let Some(p) = cands.pop() {
            return Some(p);
        }
    }
    None
}

fn nvfp4_real_ckpt_dir() -> Option<std::path::PathBuf> {
    if let Ok(d) = std::env::var("NVFP4_REAL_CKPT") {
        let p = std::path::PathBuf::from(d);
        assert!(
            p.is_dir(),
            "NVFP4_REAL_CKPT={} is not a directory",
            p.display()
        );
        return Some(p);
    }
    if let Some(p) = hub_snapshot_with(GEMMA4_31B_NVFP4_REPO, "config.json") {
        eprintln!("nvfp4_real_from_disk: using cached {}", p.display());
        return Some(p);
    }
    if std::env::var("NVFP4_REAL_ALLOW_SKIP").as_deref() == Ok("1") {
        eprintln!(
            "SKIP (NVFP4_REAL_ALLOW_SKIP=1): nvfp4_real_from_disk_forward_matches_training_dequant \
             found no {GEMMA4_31B_NVFP4_REPO} snapshot. This is a SKIP, not a pass."
        );
        return None;
    }
    panic!(
        "nvfp4_real_from_disk_forward_matches_training_dequant: NVFP4_REAL_CKPT is unset and no \
         {GEMMA4_31B_NVFP4_REPO} snapshot with config.json was found under HF_HUB_CACHE or \
         $HOME/.cache/huggingface/hub. This is the only test in this file that touches a real \
         on-disk packed NVFP4 tensor; it refuses to report success without running. Set \
         NVFP4_REAL_CKPT, or NVFP4_REAL_ALLOW_SKIP=1 to skip on purpose."
    );
}

fn fp8_linear_platform_adaptive(
    w: &Tensor,
    device: &Device,
    runner: Arc<Mutex<Fp8GemmRunner>>,
) -> Linear {
    match Linear::from_bf16_quantized_fp8(w, None, device, runner.clone()) {
        Ok(l) => l,
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(
                msg.contains("OUTER_VEC_32F"),
                "unexpected fp8 construction failure: {msg}"
            );
            eprintln!("[fp8-test] per-row refused on this platform; using tensor mode");
            Linear::from_bf16_quantized_fp8_in_mode(
                w,
                None,
                None,
                device,
                runner,
                nv_quant::fp8::Fp8ScaleMode::PerTensor,
            )
            .expect("tensor-mode fp8 must construct wherever fp8 exists")
        }
    }
}

fn rel_rms(got: &[f32], expect: &[f32]) -> f32 {
    let mut sum_sq = 0f64;
    let mut sum_expect_sq = 0f64;
    for (g, e) in got.iter().zip(expect.iter()) {
        let d = (g - e) as f64;
        sum_sq += d * d;
        sum_expect_sq += (*e as f64).powi(2);
    }
    ((sum_sq / sum_expect_sq.max(1e-12)).sqrt()) as f32
}

#[test]
fn fp8_linear_matches_bf16_linear_within_5pct() {
    use half::bf16;
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let major = detect_major(&ctx);
    if !supports_fp8(major) {
        eprintln!("skip: SM {major} lacks FP8");
        return;
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Fp8GemmRunner::new(stream.clone()).unwrap()));

    let (out_f, in_f, m) = (128usize, 128usize, 32usize);
    let w_flat: Vec<bf16> = (0..out_f * in_f)
        .map(|i| bf16::from_f32(((i as f32) * 0.013).sin()))
        .collect();
    let x_flat: Vec<bf16> = (0..m * in_f)
        .map(|i| bf16::from_f32(((i as f32) * 0.011).cos()))
        .collect();
    let w = Tensor::from_vec(w_flat, (out_f, in_f), &device).unwrap();
    let x = Tensor::from_vec(x_flat, (m, in_f), &device).unwrap();

    let bf16_lin = Linear::new(w.clone(), None).unwrap();
    let fp8_lin = fp8_linear_platform_adaptive(&w, &device, runner);

    let y_bf16 = bf16_lin.forward(&x).unwrap();
    let y_fp8 = fp8_lin.forward(&x).unwrap();
    assert_eq!(y_fp8.dims(), &[m, out_f]);
    assert_eq!(y_fp8.dtype(), DType::BF16);

    let yb: Vec<f32> = y_bf16
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let yf: Vec<f32> = y_fp8
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let r = rel_rms(&yf, &yb);
    eprintln!("fp8 linear vs bf16 ref rel rms = {r}");
    assert!(r < 0.05, "fp8 vs bf16 rel rms {r} > 0.05");
}

#[test]
fn nvfp4_linear_matches_synthetic_known_values() {
    use half::bf16;
    use nv_quant::nvfp4::{cpu_nvfp4_matmul_weight_row, Nvfp4Tensor};
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let major = detect_major(&ctx);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major} lacks NVFP4");
        return;
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream.clone()).unwrap()));

    let (m, n, k) = (128usize, 128usize, 128usize);
    let w_rows: Vec<Vec<f32>> = (0..n)
        .map(|i| (0..k).map(|j| ((i * k + j) as f32 * 0.09).cos()).collect())
        .collect();
    let x_rows: Vec<Vec<f32>> = (0..m)
        .map(|i| (0..k).map(|j| ((i * k + j) as f32 * 0.07).sin()).collect())
        .collect();
    let w_flat: Vec<f32> = w_rows.iter().flatten().copied().collect();
    let x_flat: Vec<f32> = x_rows.iter().flatten().copied().collect();
    let w_bf: Vec<bf16> = w_flat.iter().map(|x| bf16::from_f32(*x)).collect();
    let x_bf: Vec<bf16> = x_flat.iter().map(|x| bf16::from_f32(*x)).collect();
    let w_t = Tensor::from_vec(w_bf, (n, k), &device).unwrap();
    let x_t = Tensor::from_vec(x_bf, (m, k), &device).unwrap();

    let nvfp4_lin = Linear::from_bf16_quantized_nvfp4(&w_t, None, &device, runner).unwrap();
    let y = nvfp4_lin.forward(&x_t).unwrap();

    let a_q = Nvfp4Tensor::quantize_rows(&x_rows);
    let b_q = Nvfp4Tensor::quantize_rows(&w_rows);
    let expect = cpu_nvfp4_matmul_weight_row(&a_q, &b_q, m, n, k);
    let yh: Vec<f32> = y
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let mut sum_sq = 0f64;
    let mut sum_exp_sq = 0f64;
    for (g, e) in yh.iter().zip(expect.iter()) {
        let d = (g - e.to_f32()) as f64;
        sum_sq += d * d;
        sum_exp_sq += (e.to_f32() as f64).powi(2);
    }
    let r = (sum_sq / sum_exp_sq.max(1e-12)).sqrt();
    eprintln!("nvfp4 linear vs cpu dequant (synthetic sin/cos) rel rms = {r}");
    assert!(r < 0.25, "expected match to cpu dequant, got rel rms {r}");
}

#[test]
fn nvfp4_linear_matches_bf16_linear_within_18pct() {
    use half::bf16;
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let major = detect_major(&ctx);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major} lacks NVFP4");
        return;
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream.clone()).unwrap()));

    let (out_f, in_f, m) = (128usize, 128usize, 128usize);
    let w_flat: Vec<bf16> = (0..out_f * in_f)
        .map(|i| bf16::from_f32(((i as f32) * 0.07).sin()))
        .collect();
    let x_flat: Vec<bf16> = (0..m * in_f)
        .map(|i| bf16::from_f32(((i as f32) * 0.09).cos()))
        .collect();
    let w = Tensor::from_vec(w_flat, (out_f, in_f), &device).unwrap();
    let x = Tensor::from_vec(x_flat, (m, in_f), &device).unwrap();

    let bf16_lin = Linear::new(w.clone(), None).unwrap();
    let nvfp4_lin = Linear::from_bf16_quantized_nvfp4(&w, None, &device, runner).unwrap();

    let y_bf16 = bf16_lin.forward(&x).unwrap();
    let y_nvfp4 = nvfp4_lin.forward(&x).unwrap();
    assert_eq!(y_nvfp4.dims(), &[m, out_f]);
    assert_eq!(y_nvfp4.dtype(), DType::BF16);

    let yb: Vec<f32> = y_bf16
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let yn: Vec<f32> = y_nvfp4
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let r = rel_rms(&yn, &yb);
    eprintln!("nvfp4 linear vs bf16 ref rel rms = {r}");
    assert!(r < 0.18, "nvfp4 vs bf16 rel rms {r} > 0.18");
}

#[test]
fn nvfp4_linear_decode_step_m1_pad_and_trim() {
    use half::bf16;
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let major = detect_major(&ctx);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major} lacks NVFP4");
        return;
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream.clone()).unwrap()));

    let (out_f, in_f) = (128usize, 128usize);
    let w_flat: Vec<bf16> = (0..out_f * in_f)
        .map(|i| bf16::from_f32(((i as f32) * 0.07).sin()))
        .collect();
    let x_flat: Vec<bf16> = (0..in_f)
        .map(|i| bf16::from_f32(((i as f32) * 0.09).cos()))
        .collect();
    let w = Tensor::from_vec(w_flat, (out_f, in_f), &device).unwrap();
    let x = Tensor::from_vec(x_flat, (1usize, 1usize, in_f), &device).unwrap();

    let nvfp4_lin = Linear::from_bf16_quantized_nvfp4(&w, None, &device, runner).unwrap();
    let y = nvfp4_lin.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 1, out_f]);
    assert_eq!(y.dtype(), DType::BF16);

    let bf16_lin = Linear::new(w, None).unwrap();
    let y_bf16 = bf16_lin.forward(&x).unwrap();

    let yb: Vec<f32> = y_bf16
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let yn: Vec<f32> = y
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let r = rel_rms(&yn, &yb);
    eprintln!("nvfp4 m=1 vs bf16 rel rms = {r}");
    assert!(r < 0.30, "nvfp4 m=1 vs bf16 rel rms {r} > 0.30");
}

#[test]
fn nvfp4_linear_layout_isolation_check() {
    use half::bf16;
    use nv_quant::nvfp4::{cpu_nvfp4_matmul_weight_row, Nvfp4Tensor};
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let major = detect_major(&ctx);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major} lacks NVFP4");
        return;
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream.clone()).unwrap()));

    let (m, n, k) = (128usize, 128usize, 128usize);
    let x_rows: Vec<Vec<f32>> = (0..m)
        .map(|i| (0..k).map(|j| ((i * k + j) as f32 * 0.07).sin()).collect())
        .collect();
    let w_rows: Vec<Vec<f32>> = (0..n)
        .map(|i| (0..k).map(|j| ((i * k + j) as f32 * 0.09).cos()).collect())
        .collect();
    let w_flat_bf: Vec<bf16> = w_rows
        .iter()
        .flatten()
        .map(|x| bf16::from_f32(*x))
        .collect();
    let x_flat_bf: Vec<bf16> = x_rows
        .iter()
        .flatten()
        .map(|x| bf16::from_f32(*x))
        .collect();
    let w_t = Tensor::from_vec(w_flat_bf, (n, k), &device).unwrap();
    let x_t = Tensor::from_vec(x_flat_bf, (m, k), &device).unwrap();

    let lin = Linear::from_bf16_quantized_nvfp4(&w_t, None, &device, runner).unwrap();
    let y = lin.forward(&x_t).unwrap();
    let yh: Vec<f32> = y
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let x_q = Nvfp4Tensor::quantize_rows(&x_rows);
    let w_q = Nvfp4Tensor::quantize_rows(&w_rows);
    let expect = cpu_nvfp4_matmul_weight_row(&x_q, &w_q, m, n, k);
    let mut sum_sq = 0f64;
    let mut sum_exp_sq = 0f64;
    for (g, e) in yh.iter().zip(expect.iter()) {
        let d = (g - e.to_f32()) as f64;
        sum_sq += d * d;
        sum_exp_sq += (e.to_f32() as f64).powi(2);
    }
    let r = (sum_sq / sum_exp_sq.max(1e-12)).sqrt();
    eprintln!("Linear-path nvfp4 vs cpu dequant (sin/cos) rel rms = {r}");
    assert!(r < 0.18, "Linear-path should match GEMM noise");
}

#[test]
fn fp8_linear_3d_input_shape_preserved() {
    use half::bf16;
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let major = detect_major(&ctx);
    if !supports_fp8(major) {
        eprintln!("skip: SM {major} lacks FP8");
        return;
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Fp8GemmRunner::new(stream.clone()).unwrap()));

    let (out_f, in_f) = (128usize, 128usize);
    let w_flat: Vec<bf16> = (0..out_f * in_f)
        .map(|i| bf16::from_f32(((i as f32) * 0.013).sin()))
        .collect();
    let x_flat: Vec<bf16> = (0..2 * 8 * in_f)
        .map(|i| bf16::from_f32(((i as f32) * 0.011).cos()))
        .collect();
    let w = Tensor::from_vec(w_flat, (out_f, in_f), &device).unwrap();
    let x = Tensor::from_vec(x_flat, (2usize, 8usize, in_f), &device).unwrap();
    let lin = fp8_linear_platform_adaptive(&w, &device, runner);
    let y = lin.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 8, out_f]);
}

#[test]
fn nvfp4_dev_quant_linear_matches_bf16_linear() {
    use half::bf16;
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let major = detect_major(&ctx);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major} lacks NVFP4");
        return;
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream.clone()).unwrap()));

    let (out_f, in_f, m) = (256usize, 128usize, 5usize);
    let w_flat: Vec<bf16> = (0..out_f * in_f)
        .map(|i| bf16::from_f32(0.02 * ((i as f32) * 0.07).sin()))
        .collect();
    let x_flat: Vec<bf16> = (0..m * in_f)
        .map(|i| bf16::from_f32(3.0 * ((i as f32) * 0.09).cos()))
        .collect();
    let w = Tensor::from_vec(w_flat, (out_f, in_f), &device).unwrap();
    let x = Tensor::from_vec(x_flat, (m, in_f), &device).unwrap();

    let bf16_lin = Linear::new(w.clone(), None).unwrap();
    let dev_lin = Linear::from_bf16_quantized_nvfp4_dev(&w, None, &device, runner).unwrap();
    assert!(matches!(dev_lin.kind(), nv_quant::LinearKind::Nvfp4));
    assert_eq!(dev_lin.in_features(), in_f);
    assert_eq!(dev_lin.out_features(), out_f);

    let y_bf16 = bf16_lin.forward(&x).unwrap();
    let y_dev = dev_lin.forward(&x).unwrap();
    assert_eq!(y_dev.dims(), &[m, out_f]);
    assert_eq!(y_dev.dtype(), DType::BF16);

    let yb: Vec<f32> = y_bf16
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let yd: Vec<f32> = y_dev
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let r = rel_rms(&yd, &yb);
    eprintln!("nvfp4 dev-quant linear vs bf16 ref rel rms = {r}");
    assert!(r < 0.18, "nvfp4 dev-quant vs bf16 rel rms {r} > 0.18");
}

#[test]
fn nvfp4_dev_quant_rejects_bad_shapes() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let major = detect_major(&ctx);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major} lacks NVFP4");
        return;
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream.clone()).unwrap()));

    let w_small = Tensor::zeros((64usize, 128usize), DType::BF16, &device).unwrap();
    assert!(
        Linear::from_bf16_quantized_nvfp4_dev(&w_small, None, &device, runner.clone()).is_err()
    );
    let w_badk = Tensor::zeros((128usize, 136usize), DType::BF16, &device).unwrap();
    assert!(Linear::from_bf16_quantized_nvfp4_dev(&w_badk, None, &device, runner).is_err());
}

#[test]
fn nvfp4_dequant_weight_matches_served_effective_weight() {
    use half::bf16;
    use nv_quant::nvfp4::Nvfp4Tensor;
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let major = detect_major(&ctx);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major} lacks NVFP4");
        return;
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream.clone()).unwrap()));

    let (out_f, in_f, m) = (128usize, 256usize, 128usize);
    let w_f32: Vec<f32> = (0..out_f * in_f)
        .map(|i| ((i as f32) * 0.017).sin() * 0.35)
        .collect();
    let w_bf: Vec<bf16> = w_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let w = Tensor::from_vec(w_bf, (out_f, in_f), &device).unwrap();

    let lin = Linear::from_bf16_quantized_nvfp4_dev(&w, None, &device, runner).unwrap();
    let deq = lin
        .dequant_weight()
        .unwrap()
        .expect("nvfp4 dequant_weight is Some");
    assert_eq!(deq.dims(), &[out_f, in_f]);
    let deq_f32: Vec<f32> = deq
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        deq_f32.iter().all(|v| v.is_finite()),
        "dequant_weight must be finite"
    );

    let rr_grid = rel_rms(&deq_f32, &w_f32);
    println!("NVFP4_DEQUANT_VS_ORIG_RELRMS {rr_grid:.6e} (nvfp4 grid floor)");
    assert!(
        rr_grid > 0.005 && rr_grid < 0.15,
        "grid rel-RMS {rr_grid} out of band"
    );

    let amax = w_f32.iter().fold(0f32, |a, &b| a.max(b.abs()));
    let stored_global = (448.0f32 * 6.0) / amax;
    let rows: Vec<Vec<f32>> = (0..out_f)
        .map(|r| w_f32[r * in_f..(r + 1) * in_f].to_vec())
        .collect();
    let qh = Nvfp4Tensor::quantize_rows_with_global(&rows, stored_global);
    let host_eff: Vec<f32> = qh
        .dequantize_scaled(1.0 / stored_global)
        .into_iter()
        .flatten()
        .collect();
    println!(
        "NVFP4_GPU_VS_HOST_PACKER_RELRMS {:.6e} (two quantizers differ)",
        rel_rms(&deq_f32, &host_eff)
    );

    let x_f32: Vec<f32> = (0..m * in_f)
        .map(|i| ((i as f32) * 0.011).cos() * 0.5)
        .collect();
    let x_bf: Vec<bf16> = x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let x = Tensor::from_vec(x_bf, (m, in_f), &device).unwrap();
    let y_gpu: Vec<f32> = lin
        .forward(&x)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let x_rows: Vec<Vec<f32>> = (0..m)
        .map(|r| x_f32[r * in_f..(r + 1) * in_f].to_vec())
        .collect();
    let aq: Vec<Vec<f32>> = Nvfp4Tensor::quantize_rows(&x_rows).dequantize();
    let mut y_ref = vec![0f32; m * out_f];
    for i in 0..m {
        for j in 0..out_f {
            let mut acc = 0f32;
            for p in 0..in_f {
                acc += aq[i][p] * deq_f32[j * in_f + p];
            }
            y_ref[i * out_f + j] = acc;
        }
    }
    let rr_gemm = rel_rms(&y_gpu, &y_ref);
    println!("NVFP4_FORWARD_VS_DEQUANTWEIGHT_RELRMS {rr_gemm:.6e}");
    assert!(
        rr_gemm < 0.08,
        "dequant_weight is not the GEMM weight operand: forward vs deqW rel-RMS {rr_gemm}"
    );
}

#[test]
fn nvfp4_real_from_disk_forward_matches_training_dequant() {
    use half::bf16;
    use nv_layers::moe::{nvfp4_linear_from_disk_with_suffixes, Nvfp4Suffixes};
    use nv_quant::nvfp4::dequantize_packed_linear;
    use nv_weights::WeightLoader;

    let Some(dir) = nvfp4_real_ckpt_dir() else {
        return;
    };
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA");
        return;
    };
    let major = detect_major(&ctx);
    if !supports_nvfp4(major) {
        eprintln!("skip: SM {major} lacks NVFP4");
        return;
    }
    let device = Device::new_cuda(0).unwrap();
    let stream = ctx.default_stream();
    let runner = Arc::new(Mutex::new(Nvfp4GemmRunner::new(stream.clone()).unwrap()));

    let weights = WeightLoader::open_dir(&dir, &device).unwrap();
    let module = "model.language_model.layers.0.mlp.gate_proj";
    let shp = weights.shape_of(&format!("{module}.weight")).unwrap();
    let (out_f, in_f) = (shp[0], shp[1] * 2);

    let lin = nvfp4_linear_from_disk_with_suffixes(
        &weights,
        module,
        out_f,
        in_f,
        runner,
        &device,
        Nvfp4Suffixes::GEMMA_MODELOPT,
    )
    .unwrap();
    assert_eq!(lin.in_features(), in_f);
    assert_eq!(lin.out_features(), out_f);

    let packed = weights
        .raw_bytes(&format!("{module}.weight"))
        .unwrap()
        .to_vec();
    let scales = weights
        .raw_bytes(&format!("{module}.weight_scale"))
        .unwrap()
        .to_vec();
    let ws2: f32 = weights
        .get(&format!("{module}.weight_scale_2"), DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()[0];
    let input_scale: f32 = weights
        .get(&format!("{module}.input_scale"), DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()[0];
    let w_eff = dequantize_packed_linear(&packed, &scales, out_f, in_f, ws2);

    let deqw: Vec<f32> = lin
        .dequant_weight()
        .unwrap()
        .expect("Some")
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let ratio_first = if w_eff[100 * in_f + 100].abs() > 0.0 {
        deqw[100 * in_f + 100] / w_eff[100 * in_f + 100]
    } else {
        f32::NAN
    };
    println!(
        "REAL_DEQUANTWEIGHT_VS_TRAINEFF ws2={ws2:.6e} input_scale={input_scale:.6e} \
         deqw/w_eff@[100,100]={ratio_first:.6e} (expect ~= input_scale)"
    );

    let w_bf: Vec<bf16> = w_eff.iter().map(|&v| bf16::from_f32(v)).collect();
    let w_t = Tensor::from_vec(w_bf, (out_f, in_f), &device).unwrap();
    let dense = Linear::new(w_t, None).unwrap();

    let m = 64usize;
    let x_flat: Vec<bf16> = (0..m * in_f)
        .map(|i| bf16::from_f32(((i as f32) * 0.013).sin() * 0.7))
        .collect();
    let x = Tensor::from_vec(x_flat, (m, in_f), &device).unwrap();

    let y_gpu: Vec<f32> = lin
        .forward(&x)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let y_dense: Vec<f32> = dense
        .forward(&x)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let rr = rel_rms(&y_gpu, &y_dense);
    println!("REAL_NVFP4_FORWARD_VS_TRAINDEQUANT_RELRMS {rr:.6e} (m={m}, {module})");
    assert!(
        rr < 0.10,
        "real serving NVFP4 GEMM does not match training-dequant dense forward: rel-RMS {rr}"
    );
    println!("REAL_GPU_EQUIV_OK");
}
