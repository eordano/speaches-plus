#![cfg(feature = "wgpu")]

mod common;
use common::q3d_lora_synth::{synth_adapter, write_adapter, Fill, PREFIX};

use nv_models::dots_ocr::decoder_wgpu::dots_decoder_dense_config_from_hf_json_str;
use nv_models::qwen3_5_dense_wgpu::Q3dLora;

const DOTS_TINY_CONFIG: &str = r#"{
    "model_type": "dots_ocr",
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "vocab_size": 128,
    "max_position_embeddings": 512,
    "eos_token_id": 5
}"#;

const DOTS_TRAINER_PREFIX: &str = "language_model.model.layers";

#[test]
fn a_dots_trained_adapter_lands_on_every_q3d_attention_and_mlp_site() {
    let cfg = dots_decoder_dense_config_from_hf_json_str(DOTS_TINY_CONFIG).expect("dots config");
    let mut synth = synth_adapter(&cfg, 0xd075, Fill::Coarse, None);
    synth.mods = synth
        .mods
        .into_iter()
        .map(|(name, ab)| (name.replace(PREFIX, DOTS_TRAINER_PREFIX), ab))
        .collect();
    for name in synth.mods.keys() {
        assert!(
            name.starts_with(DOTS_TRAINER_PREFIX),
            "synth module {name} did not take the dots trainer spelling"
        );
    }
    let dir = write_adapter("dots-names", &synth);
    let lora = Q3dLora::from_peft_dir(&dir.0, &cfg)
        .expect("a dots-spelled adapter must build q3d serving sites");
    assert_eq!(
        lora.matched_modules(),
        7 * cfg.num_hidden_layers,
        "every q/k/v/o/gate/up/down of every layer must match; skipped: {:?}",
        lora.skipped_modules()
    );
    assert!(
        lora.skipped_modules().is_empty(),
        "no dots module may be skipped: {:?}",
        lora.skipped_modules()
    );
    std::fs::remove_dir_all(&dir.0).ok();
}
