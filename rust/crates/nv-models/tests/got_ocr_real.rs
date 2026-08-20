#![cfg(feature = "cuda")]

mod common;
use common::cer;
use common::fixture_expected_text;
use common::fixture_rgb;
mod hub_snapshot;

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device};
use nv_models::deepseek_ocr::RgbImage;
use nv_models::dots_ocr::DotsDecoderConfig;
use nv_models::got_ocr::pipeline::{GotMode, GotOcrPipeline, GOT_IMAGE_TOKENS};
use nv_models::got_ocr::vision::sam_checkpoint_name;
use common::got_ocr_snapshot_dir as snapshot_dir;

const GOT_CONFIG_JSON: &str = r#"{
  "model_type": "got_ocr2",
  "image_seq_length": 576,
  "image_token_index": 151859,
  "torch_dtype": "bfloat16",
  "text_config": {
    "model_type": "qwen2",
    "hidden_size": 1024,
    "intermediate_size": 2816,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "vocab_size": 151860,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-06,
    "max_position_embeddings": 32768,
    "tie_word_embeddings": true
  }
}"#;

const GOT_CER_GATE_MEASURED_0_0000_ON_070_AND_071_CUDA_BF16_D3017EF: f64 = 0.05;

#[test]
fn sam_checkpoint_name_maps_every_representative_tensor() {
    assert_eq!(
        sam_checkpoint_name("patch_embed.proj.weight"),
        "vision_tower.patch_embed.projection.weight"
    );
    assert_eq!(
        sam_checkpoint_name("blocks.3.norm1.weight"),
        "vision_tower.layers.3.layer_norm1.weight"
    );
    assert_eq!(
        sam_checkpoint_name("blocks.3.norm2.bias"),
        "vision_tower.layers.3.layer_norm2.bias"
    );
    assert_eq!(
        sam_checkpoint_name("blocks.3.attn.qkv.bias"),
        "vision_tower.layers.3.attn.qkv.bias"
    );
    assert_eq!(
        sam_checkpoint_name("blocks.3.attn.proj.weight"),
        "vision_tower.layers.3.attn.proj.weight"
    );
    assert_eq!(
        sam_checkpoint_name("blocks.3.attn.rel_pos_h"),
        "vision_tower.layers.3.attn.rel_pos_h"
    );
    assert_eq!(
        sam_checkpoint_name("blocks.3.mlp.lin1.weight"),
        "vision_tower.layers.3.mlp.lin1.weight"
    );
    assert_eq!(sam_checkpoint_name("pos_embed"), "vision_tower.pos_embed");
}

#[test]
fn config_parses_and_group_counts_sum_to_the_pinned_tensor_count() {
    let cfg = nv_ocr::ModelOcrConfig::from_json_str(GOT_CONFIG_JSON).expect("parse config");
    let groups = nv_ocr::expected_weight_groups(&cfg);
    let total: usize = groups.iter().map(|g| g.expected).sum();
    assert_eq!(
        total,
        nv_ocr::WEIGHT_TENSOR_COUNT,
        "expected_weight_groups must account for all {} tensors",
        nv_ocr::WEIGHT_TENSOR_COUNT
    );
    let v: serde_json::Value = serde_json::from_str(GOT_CONFIG_JSON).unwrap();
    let text = DotsDecoderConfig::from_hf_json_str(&v["text_config"].to_string()).expect("text cfg");
    assert_eq!(text.head_dim(), 64);
    assert!(text.attention_bias);
    assert!(text.tie_word_embeddings);
    assert_eq!(text.hidden_size, 1024);
}

#[test]
#[ignore]
fn got_ocr2_full_pipeline_reads_a_rendered_page() {
    if std::env::var("NV_GOT_OCR_TEST").as_deref() != Ok("1") {
        hub_snapshot::precondition_absent(
            "got_ocr2_full_pipeline_reads_a_rendered_page",
            "NV_GOT_OCR_TEST=1 opt-in",
            "set NV_GOT_OCR_TEST=1 and NV_GOT_OCR_DIR to the GOT-OCR-2.0-hf snapshot",
        );
        return;
    }
    let dir = snapshot_dir().expect("GOT-OCR-2.0-hf snapshot present");
    let device = Device::new_cuda(0).expect("cuda device 0");
    let t0 = Instant::now();
    let pipeline = GotOcrPipeline::load(&dir, &device).expect("load GOT-OCR2 pipeline");

    let feats = pipeline
        .encode_image(&fixture_rgb("070-ocr-paragraph"))
        .expect("vision forward");
    assert_eq!(feats.dims2().unwrap(), (GOT_IMAGE_TOKENS, 1024));
    let host: Vec<f32> = feats
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(host.iter().all(|v| v.is_finite()), "vision output has non-finite values");
    let mean: f64 = host.iter().map(|&v| v as f64).sum::<f64>() / host.len() as f64;
    let var: f64 =
        host.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / host.len() as f64;
    assert!(var.sqrt() > 1e-3, "vision output is degenerate (std {})", var.sqrt());
    let tok = |i: usize| &host[i * 1024..(i + 1) * 1024];
    for (a, b) in [(0usize, 100usize), (0, 255), (100, 255)] {
        assert!(
            tok(a).iter().zip(tok(b)).any(|(x, y)| x != y),
            "vision tokens {a} and {b} are identical"
        );
    }

    let plain = pipeline
        .recognize(&fixture_rgb("070-ocr-paragraph"), GotMode::Plain, 512)
        .expect("recognize plain");
    assert!(plain.hit_eos || plain.generated_tokens > 0, "no tokens generated");
    assert!(
        plain.text.to_lowercase().contains("the quick brown fox"),
        "decoded text missing rendered words: {:?}",
        plain.text
    );

    let mut worst = 0.0f64;
    for name in ["070-ocr-paragraph", "071-ocr-layout-letter"] {
        let res = pipeline
            .recognize(&fixture_rgb(name), GotMode::Plain, 1024)
            .expect("recognize");
        let want = fixture_expected_text(name);
        let e = cer(&res.text, &want);
        eprintln!(
            "[real] got-ocr2 CER on {name} = {e:.4} (basis: snapshot {}, cuda bf16, 1024 max_new, whitespace-collapsed lowercase vs fixture expected_text)",
            dir.display()
        );
        worst = worst.max(e);
    }
    assert!(
        worst < GOT_CER_GATE_MEASURED_0_0000_ON_070_AND_071_CUDA_BF16_D3017EF,
        "worst CER {worst:.4} exceeds the pinned gate {GOT_CER_GATE_MEASURED_0_0000_ON_070_AND_071_CUDA_BF16_D3017EF}"
    );
    eprintln!("[real] got-ocr2 full pipeline elapsed {:?}", t0.elapsed());
}
