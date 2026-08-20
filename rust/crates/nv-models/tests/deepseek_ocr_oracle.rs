use candle_core::{DType, Device, Tensor};
use nv_models::deepseek_ocr::preprocess::{prepare, ResolutionMode, RgbImage};
use nv_models::deepseek_ocr::{DeepSeekOcr2Vision, VisionConfig};
use nv_weights::WeightLoader;
use std::path::{Path, PathBuf};

fn env_dir(key: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| default.to_string()))
}

fn home_dir() -> String {
    std::env::var("HOME").unwrap_or_default()
}

fn model_dir() -> PathBuf {
    env_dir(
        "NV_DSOCR2_MODEL_DIR",
        &format!(
            "{}/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR-2/snapshots/aaa02f3811945a91062062994c5c4a3f4c0af2b0",
            home_dir()
        ),
    )
}

fn oracle_dir() -> PathBuf {
    env_dir(
        "NV_DSOCR2_ORACLE_DIR",
        &format!("{}/tmp/ocr-round/oracle", home_dir()),
    )
}

fn read_bin(dir: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap())
            .unwrap();
    let shape: Vec<usize> = meta["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let bytes = std::fs::read(dir.join(format!("{name}.bin"))).unwrap();
    let vals: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(vals.len(), shape.iter().product::<usize>());
    (vals, shape)
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, w) in got.iter().zip(want) {
        num += ((g - w) as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
}

fn tensor_of(vals: &[f32], shape: &[usize], dev: &Device) -> Tensor {
    Tensor::from_slice(vals, shape, dev).unwrap()
}

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if std::env::var("NV_DSOCR2_CPU").is_err() {
            if let Ok(d) = Device::new_cuda(0) {
                return d;
            }
        }
    }
    Device::Cpu
}

fn synth_image(w: usize, h: usize) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        [
            ((7 * x + 13 * y) % 256) as u8,
            ((3 * x + 29 * y) % 256) as u8,
            ((11 * x + 5 * y + 128) % 256) as u8,
        ]
    })
}

fn pin_deskew_off_oracle_is_hf_reference_preprocessing_which_never_rotates() {
    std::env::set_var("NV_DSOCR_DESKEW", "0");
}

#[test]
#[ignore]
fn oracle_preprocess_parity() {
    pin_deskew_off_oracle_is_hf_reference_preprocessing_which_never_rotates();
    let dir = oracle_dir();
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let w = meta["width"].as_u64().unwrap() as usize;
    let h = meta["height"].as_u64().unwrap() as usize;
    let img = synth_image(w, h);
    let prep = prepare(&img, ResolutionMode::Gundam).unwrap();
    let grid: Vec<usize> = meta["crop_grid"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(
        prep.crop_grid,
        (grid[0], grid[1]),
        "Gundam crop grid diverged from the oracle; the synthetic diagonal-stripe image trips the skew estimator, so this test runs with NV_DSOCR_DESKEW=0"
    );
    assert_eq!(
        prep.tiles.len(),
        meta["num_tiles"].as_u64().unwrap() as usize
    );

    let (want_g, _) = read_bin(&dir, "prep_global");
    let g_rel = rel_l2(&prep.global, &want_g);
    let g_max = max_abs(&prep.global, &want_g);
    println!("prep_global rel_l2={g_rel:.3e} max_abs={g_max:.4}");

    let (want_t, tshape) = read_bin(&dir, "prep_tiles");
    let mut got_t = Vec::new();
    for t in &prep.tiles {
        got_t.extend_from_slice(t);
    }
    assert_eq!(got_t.len(), tshape.iter().product::<usize>());
    let t_rel = rel_l2(&got_t, &want_t);
    let t_max = max_abs(&got_t, &want_t);
    println!("prep_tiles rel_l2={t_rel:.3e} max_abs={t_max:.4}");

    assert!(g_rel < 5e-3, "global preprocess rel_l2 {g_rel}");
    assert!(t_rel < 5e-3, "tile preprocess rel_l2 {t_rel}");
    assert!(g_max < 0.05, "global preprocess max_abs {g_max}");
    assert!(t_max < 0.05, "tile preprocess max_abs {t_max}");
}

#[test]
#[ignore]
fn oracle_encoder_parity_f32() {
    let dir = oracle_dir();
    let dev = device();
    let (dtype, tol) = match std::env::var("NV_DSOCR2_DTYPE").as_deref() {
        Ok("bf16") => (DType::BF16, 20.0f64),
        _ => (DType::F32, 1.0f64),
    };
    let flow_tol = match dtype {
        DType::BF16 => 0.12f64,
        _ => 5e-3,
    };
    println!("device: {:?} dtype: {:?}", dev, dtype);
    let shard = model_dir().join("model-00001-of-000001.safetensors");
    let weights = WeightLoader::open_file(&shard, &dev).unwrap();
    let vision = DeepSeekOcr2Vision::from_loader(
        &weights,
        "model.",
        VisionConfig::deepseek_ocr2(),
        &dev,
        dtype,
    )
    .unwrap();

    let (pg, pg_shape) = read_bin(&dir, "prep_global");
    let global = tensor_of(&pg, &pg_shape, &dev).unsqueeze(0).unwrap();
    let (pt, pt_shape) = read_bin(&dir, "prep_tiles");
    let tiles = tensor_of(&pt, &pt_shape, &dev);

    let sam_g = vision.sam_compress(&global).unwrap();
    let (want, want_shape) = read_bin(&dir, "sam_global");
    assert_eq!(sam_g.dims(), want_shape.as_slice());
    let rel = rel_l2(&to_vec(&sam_g), &want);
    println!("sam_global rel_l2={rel:.3e}");
    assert!(rel < 2e-3 * tol, "sam_global rel_l2 {rel}");

    let sam_t = vision.sam_compress(&tiles).unwrap();
    let (want, want_shape) = read_bin(&dir, "sam_tiles");
    assert_eq!(sam_t.dims(), want_shape.as_slice());
    let rel = rel_l2(&to_vec(&sam_t), &want);
    println!("sam_tiles rel_l2={rel:.3e}");
    assert!(rel < 5e-3 * tol, "sam_tiles rel_l2 {rel}");

    let flow_g = vision.flow_features(&sam_g).unwrap();
    let (want, want_shape) = read_bin(&dir, "flow_global");
    assert_eq!(flow_g.dims(), want_shape.as_slice());
    let rel = rel_l2(&to_vec(&flow_g), &want);
    println!("flow_global rel_l2={rel:.3e}");
    assert!(rel < flow_tol, "flow_global rel_l2 {rel}");

    let flow_t = vision.flow_features(&sam_t).unwrap();
    let (want, want_shape) = read_bin(&dir, "flow_tiles");
    assert_eq!(flow_t.dims(), want_shape.as_slice());
    let rel = rel_l2(&to_vec(&flow_t), &want);
    println!("flow_tiles rel_l2={rel:.3e}");
    assert!(rel < flow_tol, "flow_tiles rel_l2 {rel}");

    let feats = vision.encode_views(&global, Some(&tiles)).unwrap();
    let (want, want_shape) = read_bin(&dir, "features");
    assert_eq!(feats.dims(), want_shape.as_slice());
    let got = to_vec(&feats);
    let rel = rel_l2(&got, &want);
    let mabs = max_abs(&got, &want);
    println!("features rel_l2={rel:.3e} max_abs={mabs:.4e}");
    assert!(rel < flow_tol, "features rel_l2 {rel}");
}
