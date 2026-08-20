#![cfg(all(feature = "cuda", feature = "wgpu"))]

use candle_core::{DType, Device, Tensor};
use nv_models::embed_row_splice::rows_to_bf16;
use nv_models::qwen3_5_dense_wgpu::{ImageRowSplice, Qwen3_5DenseWgpu};
use nv_models::qwen3_5_moe::Qwen3_5DenseConfig;
use nv_models::qwen3_mm_splice::{
    build_mrope_positions_matching_hf_get_rope_index, mrope_section_from_hf_json_str,
    QWEN3_5_IMAGE_TOKEN_ID_FROM_THE_RELEASE_CONFIG,
};
use nv_omni::{Qwen3VisionConfig, Qwen3VisionTower};
use nv_weights::WeightLoader;
use std::path::PathBuf;
use std::time::Instant;

const REPO: &str = "unsloth/Qwen3.8-27B-NVFP4";
const GATE_ENV: &str = "NV_Q38_VISION_TEST";
const IMAGE_SIDE_448_GIVES_A_28X28_PATCH_GRID_MERGING_TO_196_TOKENS: u32 = 448;
const MERGED_GRID_14X14: (usize, usize, usize) = (1, 14, 14);

fn require_gate() {
    if std::env::var(GATE_ENV).as_deref() != Ok("1") {
        panic!("set {GATE_ENV}=1 to run the real-weights batched-vs-walk image splice parity");
    }
}

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
        if !d.is_empty() {
            let p = PathBuf::from(&d);
            assert!(p.join("config.json").exists(), "NV_QWEN38_DIR={d} has no config.json");
            return p;
        }
    }
    let root = PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", REPO.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(&root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists() && p.join("model.safetensors").exists())
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
    img
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

fn real_image_rows_bf16_from_the_cuda_tower(dir: &PathBuf) -> Vec<u16> {
    let device = Device::new_cuda_with_stream(0).expect("cuda device for the vision tower");
    let weights = WeightLoader::open_dir(dir, &device).expect("open snapshot");
    let vis_cfg = Qwen3VisionConfig::from_hf_config_json(dir.join("config.json"))
        .expect("vision_config parses");
    let mut tower = Qwen3VisionTower::new_empty(vis_cfg, &device).expect("tower");
    tower.load_weights(&weights).expect("visual.* bf16 tensors load");
    let pixels = normalized_pixel_tensor(&test_png_rgb(), &device);
    let rows = tower.forward(&pixels).expect("tower forward");
    let (n_rows, width) = rows.dims2().expect("2d embedding");
    assert_eq!((n_rows, width), (196, 5120), "448x448 merges to 196 trunk rows");
    let flat: Vec<f32> = rows
        .to_dtype(DType::F32)
        .expect("f32 rows")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("host rows");
    assert!(flat.iter().all(|v| v.is_finite()), "image rows finite");
    rows_to_bf16(&flat)
}

fn prompt_ids_with_196_image_slots(dir: &PathBuf) -> (Vec<u32>, usize) {
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
    assert_eq!(
        img_positions[195],
        img_positions[0] + 195,
        "image slots form one contiguous run"
    );
    assert_ne!(
        *ids.last().expect("non-empty prompt"),
        QWEN3_5_IMAGE_TOKEN_ID_FROM_THE_RELEASE_CONFIG,
        "the assistant header guarantees a text final token"
    );
    (ids, img_positions[0])
}

#[test]
#[ignore = "boots the bf16 vision tower on cuda plus the ~22.6 GB NVFP4 decoder on wgpu, then \
            prefills the same real-image prompt twice: once through the batched M-row splice \
            path and once as a per-token walk through the same graph; argmax of the first \
            generated token must match; set NV_Q38_VISION_TEST=1"]
fn batched_image_splice_prefill_matches_the_per_token_walk_argmax_on_a_real_image() {
    require_gate();
    let dir = snapshot_dir();
    let rows_bf16 = real_image_rows_bf16_from_the_cuda_tower(&dir);
    let (ids, first_img_pos) = prompt_ids_with_196_image_slots(&dir);
    let hidden = 5120usize;

    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("dense config");
    let section = mrope_section_from_hf_json_str(&raw_cfg).expect("mrope sections");
    let mrope = build_mrope_positions_matching_hf_get_rope_index(
        &ids,
        QWEN3_5_IMAGE_TOKEN_ID_FROM_THE_RELEASE_CONFIG,
        &[MERGED_GRID_14X14],
    )
    .expect("mrope positions");
    assert!(!mrope.is_text_degenerate());

    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated parity must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let loader = WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("open weights");
    let max_seq = ids.len() + 8;
    let mut m = Qwen3_5DenseWgpu::from_loader(cfg, &loader, max_seq)
        .expect("Qwen3_5DenseWgpu::from_loader on the real checkpoint");
    drop(loader);
    let chunk_m = m.prefill_chunk_len();
    assert!(
        chunk_m >= 2,
        "this parity needs the chunked prefill graph (NV_WGPU_PREFILL_M >= 2)"
    );

    let (last, rest) = ids.split_last().expect("non-empty prompt");
    let splices = vec![ImageRowSplice {
        position: first_img_pos,
        rows_bf16: rows_bf16.clone(),
    }];

    m.reset().expect("reset before the batched arm");
    m.install_mrope_rows_for_prompt_and_shift_continuation_rows_by_delta(&mrope, section)
        .expect("mrope install for the batched arm");
    let t_batched = Instant::now();
    let done = m
        .prefill_tokens_with_image_rows(rest, &splices)
        .expect("batched spliced prefill");
    assert_eq!(done, rest.len(), "batched prefill must consume every prompt token");
    let (arg_batched, logits_batched) =
        m.decode_step_logits(*last).expect("batched last step");
    let batched_s = t_batched.elapsed().as_secs_f64();

    m.reset().expect("reset before the walk arm");
    m.install_mrope_rows_for_prompt_and_shift_continuation_rows_by_delta(&mrope, section)
        .expect("mrope install for the walk arm");
    let t_walk = Instant::now();
    for (i, t) in rest.iter().enumerate() {
        let in_image = i >= first_img_pos && i < first_img_pos + 196;
        let one = if in_image {
            let slot = i - first_img_pos;
            vec![ImageRowSplice {
                position: 0,
                rows_bf16: rows_bf16[slot * hidden..(slot + 1) * hidden].to_vec(),
            }]
        } else {
            Vec::new()
        };
        let done = m
            .prefill_tokens_with_image_rows(&[*t], &one)
            .expect("per-token spliced prefill");
        assert_eq!(done, 1, "the walk must consume exactly one token per call");
    }
    let (arg_walk, logits_walk) = m.decode_step_logits(*last).expect("walk last step");
    let walk_s = t_walk.elapsed().as_secs_f64();

    let max_abs = logits_batched
        .iter()
        .zip(&logits_walk)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "[q38-img-splice-parity] prompt_tokens={} image_rows=196 chunk_m={chunk_m} \
         coop={} batched_prefill_s={batched_s:.2} walk_prefill_s={walk_s:.2} \
         speedup={:.1}x argmax_batched={arg_batched} argmax_walk={arg_walk} \
         logits_max_abs={max_abs:.3e}",
        ids.len(),
        std::env::var("NV_Q3D_PF_COOP").as_deref() == Ok("1"),
        walk_s / batched_s
    );
    assert_eq!(
        arg_batched, arg_walk,
        "the batched image-splice prefill and the per-token walk disagree on the first \
         generated token"
    );
}
