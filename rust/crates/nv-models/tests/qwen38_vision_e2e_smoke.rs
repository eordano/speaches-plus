#![cfg(feature = "cuda")]

mod hub_snapshot;

use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_models::qwen3_mm_splice::{
    build_mrope_positions_matching_hf_get_rope_index, mrope_section_from_hf_json_str,
    Qwen3ImageRowSplice, QWEN3_5_IMAGE_TOKEN_ID_FROM_THE_RELEASE_CONFIG,
};
use nv_omni::{Qwen3VisionConfig, Qwen3VisionTower};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;

const REPO: &str = "unsloth/Qwen3.8-27B-NVFP4";
const GATE_ENV: &str = "NV_Q38_VISION_TEST";
const IMAGE_SIDE_448_GIVES_A_28X28_PATCH_GRID_MERGING_TO_196_TOKENS: u32 = 448;
const MERGED_GRID_14X14: (usize, usize, usize) = (1, 14, 14);
const NEW_TOKENS_32: usize = 32;
const MIN_DISTINCT_8_A_DEGENERATE_LOOP_REPEATS_ONE_OR_TWO_TOKENS: usize = 8;

fn require_gate() {
    if std::env::var(GATE_ENV).as_deref() != Ok("1") {
        panic!("set {GATE_ENV}=1 to run the real-weights qwen3.8 vision end-to-end smoke");
    }
}

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
        if !d.is_empty() {
            return PathBuf::from(d);
        }
    }
    hub_snapshot::snapshot_of(REPO, &["config.json", "*.safetensors"])
        .unwrap_or_else(|| panic!("no hydrated {REPO} snapshot; set NV_QWEN38_DIR"))
}

fn test_png_rgb() -> image::RgbImage {
    let side = IMAGE_SIDE_448_GIVES_A_28X28_PATCH_GRID_MERGING_TO_196_TOKENS;
    let mut img = image::RgbImage::new(side, side);
    for y in 0..side {
        for x in 0..side {
            let px = if x < side / 2 {
                image::Rgb([200u8, 30, 30])
            } else {
                image::Rgb([
                    (x * 255 / side) as u8,
                    (y * 255 / side) as u8,
                    ((x + y) * 255 / (2 * side)) as u8,
                ])
            };
            img.put_pixel(x, y, px);
        }
    }
    let mut png_bytes: Vec<u8> = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut png_bytes),
        image::ImageFormat::Png,
    )
    .expect("encode png");
    let decoded = image::load_from_memory(&png_bytes).expect("decode png back");
    decoded.to_rgb8()
}

fn normalized_pixel_tensor(img: &image::RgbImage, device: &Device) -> Tensor {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut chw = vec![0f32; 3 * h * w];
    for (x, y, px) in img.enumerate_pixels() {
        for c in 0..3 {
            chw[c * h * w + y as usize * w + x as usize] = (px.0[c] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
    Tensor::from_vec(chw, (1, 3, h, w), device).expect("pixel tensor")
}

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 dense arm plus its bf16 vision tower, encodes a \
            generated PNG, splices 196 image tokens at 248056 with the real 3D interleaved \
            mrope, and greedily decodes 32 tokens; set NV_Q38_VISION_TEST=1"]
fn qwen38_vision_encode_splice_generate_smoke() {
    require_gate();
    std::env::set_var(
        "NV_Q38_FUSED_QKV",
        "0",
    );
    let dir = snapshot_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");

    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let dense_cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("quant config");
    let section = mrope_section_from_hf_json_str(&raw_cfg).expect("mrope sections");
    assert_eq!(section, [11, 11, 10]);

    let weights = WeightLoader::open_dir(&dir, &device).expect("open snapshot");
    let vis_cfg = Qwen3VisionConfig::from_hf_config_json(dir.join("config.json"))
        .expect("vision_config parses");
    let mut tower = Qwen3VisionTower::new_empty(vis_cfg, &device).expect("tower");
    tower.load_weights(&weights).expect("visual.* bf16 tensors load");
    let model = Qwen3Moe::from_loader_dense_quantized(dense_cfg.clone(), &weights, &qcfg, &device)
        .expect("27B dense arm loads");
    drop(weights);

    let img = test_png_rgb();
    let pixels = normalized_pixel_tensor(&img, &device);
    device.synchronize().expect("sync before encode");
    let t_enc = std::time::Instant::now();
    let image_rows = tower.forward(&pixels).expect("tower forward");
    device.synchronize().expect("sync after encode");
    let encode_ms = t_enc.elapsed().as_secs_f64() * 1e3;
    let (n_rows, width) = image_rows.dims2().expect("2d embedding");
    assert_eq!((n_rows, width), (196, 5120), "448x448 merges to 196 trunk rows");
    let flat: Vec<f32> = image_rows
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "image rows finite");

    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .expect("tokenizer.json ships with the checkpoint");
    let prompt = format!(
        "<|im_start|>user\n<|vision_start|>{}<|vision_end|>Describe this image in one \
         sentence.<|im_end|>\n<|im_start|>assistant\n",
        "<|image_pad|>".repeat(196)
    );
    let ids: Vec<u32> = tokenizer
        .encode(prompt.as_str(), false)
        .expect("encode prompt")
        .get_ids()
        .to_vec();
    let img_positions: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == QWEN3_5_IMAGE_TOKEN_ID_FROM_THE_RELEASE_CONFIG)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(img_positions.len(), 196, "the prompt reserves 196 image slots");
    let first = img_positions[0];
    assert_eq!(
        img_positions[195],
        first + 195,
        "image slots form one contiguous run"
    );

    let mrope = build_mrope_positions_matching_hf_get_rope_index(
        &ids,
        QWEN3_5_IMAGE_TOKEN_ID_FROM_THE_RELEASE_CONFIG,
        &[MERGED_GRID_14X14],
    )
    .expect("mrope positions");
    assert!(!mrope.is_text_degenerate());

    let seq = ids.len();
    let mut cache = model.new_kv_cache(seq + NEW_TOKENS_32 + 8).expect("kv cache");
    let t_dev = Tensor::from_vec(ids.clone(), (1usize, seq), &device).expect("tokens");
    let splices = [Qwen3ImageRowSplice {
        position: first,
        rows: image_rows,
    }];
    let t_pf = std::time::Instant::now();
    let logits = model
        .forward_with_cache_prefill_image_rows_last_row_logits(
            &t_dev, &splices, &mrope, section, &mut cache,
        )
        .expect("image prefill");
    device.synchronize().expect("sync after prefill");
    let prefill_ms = t_pf.elapsed().as_secs_f64() * 1e3;

    let argmax = |logits: &Tensor| -> u32 {
        let v: Vec<f32> = logits
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap()
    };

    let mut generated: Vec<u32> = vec![argmax(&logits)];
    let t_dec = std::time::Instant::now();
    for step in 0..NEW_TOKENS_32 - 1 {
        let tok = *generated.last().unwrap();
        let token_index = seq + step;
        let pos = mrope.decode_position(token_index);
        assert!(pos >= 0, "decode position stays non-negative");
        let t1 = Tensor::from_vec(vec![tok], (1usize, 1), &device).unwrap();
        let p1 = Tensor::from_vec(vec![pos as i32], 1usize, &device).unwrap();
        let logits = model
            .forward_with_cache(&t1, &p1, &mut cache)
            .expect("decode step");
        generated.push(argmax(&logits));
    }
    device.synchronize().expect("sync after decode");
    let decode_ms_per_tok =
        t_dec.elapsed().as_secs_f64() * 1e3 / (NEW_TOKENS_32 as f64 - 1.0);

    let distinct: std::collections::BTreeSet<u32> = generated.iter().copied().collect();
    let text = tokenizer.decode(&generated, true).unwrap_or_default();
    eprintln!(
        "[q38-vision-e2e] basis: {REPO} cuda bf16-tower + nvfp4-trunk, 448x448 png, 196 image \
         tokens at {first}, seq={seq}; encode={encode_ms:.1} ms, prefill={prefill_ms:.1} ms, \
         decode={decode_ms_per_tok:.1} ms/tok; distinct={}/{}; text={text:?}",
        distinct.len(),
        generated.len()
    );
    assert!(
        distinct.len() >= MIN_DISTINCT_8_A_DEGENERATE_LOOP_REPEATS_ONE_OR_TWO_TOKENS,
        "degenerate generation: {} distinct tokens in {generated:?} (text {text:?})",
        distinct.len()
    );
    assert!(
        generated
            .iter()
            .all(|t| (*t as usize) < dense_cfg.vocab_size),
        "generated ids stay in vocab"
    );
}
