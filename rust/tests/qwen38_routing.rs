#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use speaches_plus::oapi::backend_select::QWEN35_DENSE_NO_CUDA;
use speaches_plus::oapi::chat_engine::NvEngineChat;

const DENSE_OPT_IN_ENV: &str = "NV_QWEN35_DENSE_CUDA_SERVE";

fn three_linear_then_one_full(n: usize) -> String {
    (0..n)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                "\"full_attention\""
            } else {
                "\"linear_attention\""
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn qwen38_27b_config_json() -> String {
    format!(
        r#"{{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "tie_word_embeddings": false,
  "transformers_version": "4.57.3",
  "text_config": {{
    "model_type": "qwen3_5_text",
    "hidden_size": 5120,
    "num_hidden_layers": 48,
    "num_attention_heads": 20,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "intermediate_size": 17408,
    "full_attention_interval": 4,
    "mamba_ssm_dtype": "float32",
    "output_gate_type": "swish",
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "attn_output_gate": true,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "eos_token_id": 248044,
    "rope_parameters": {{"rope_theta": 10000000.0, "partial_rotary_factor": 0.25, "rope_type": "default"}},
    "layer_types": [{}]
  }},
  "vision_config": {{"hidden_size": 1152, "depth": 27, "patch_size": 16, "temporal_patch_size": 2}}
}}"#,
        three_linear_then_one_full(48)
    )
}

fn qwen38_flagship_config_json() -> String {
    format!(
        r#"{{
  "architectures": ["Qwen3_5MoeForCausalLM"],
  "model_type": "qwen3_5_moe_text",
  "attention_bias": false,
  "attn_output_gate": true,
  "output_gate_type": "swish",
  "mamba_ssm_dtype": "float32",
  "full_attention_interval": 4,
  "head_dim": 256,
  "hidden_size": 8192,
  "num_hidden_layers": 96,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "num_experts": 512,
  "num_experts_per_tok": 10,
  "moe_intermediate_size": 2048,
  "shared_expert_intermediate_size": 2048,
  "router_aux_loss_coef": 0.001,
  "mtp_num_hidden_layers": 1,
  "mtp_use_dedicated_embeddings": true,
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 128,
  "linear_key_head_dim": 128,
  "linear_value_head_dim": 128,
  "linear_conv_kernel_dim": 4,
  "max_position_embeddings": 262144,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {{"partial_rotary_factor": 0.25, "rope_theta": 10000000.0, "rope_type": "default"}},
  "tie_word_embeddings": false,
  "transformers_version": "4.57.3",
  "vocab_size": 248320,
  "layer_types": [{}]
}}"#,
        three_linear_then_one_full(96)
    )
}

fn stub_checkpoint_dir(tag: &str, config: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qwen38-routing-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create stub checkpoint dir");
    std::fs::write(dir.join("config.json"), config).expect("write config.json");
    std::fs::write(dir.join("tokenizer.json"), "{}").expect("write tokenizer.json stub");
    std::fs::write(dir.join("model.safetensors"), b"stub").expect("write safetensors stub");
    dir
}

fn load_error(dir: &Path) -> String {
    format!(
        "{:#}",
        NvEngineChat::try_load(dir)
            .map(|_| ())
            .expect_err("a stub checkpoint with 4-byte safetensors must never finish loading")
    )
}

#[test]
fn the_27b_shape_is_refused_with_the_qwen35_dense_message_and_the_opt_in_reaches_the_loader() {
    let dir = stub_checkpoint_dir("dense", &qwen38_27b_config_json());
    std::env::remove_var(DENSE_OPT_IN_ENV);
    let refusal = load_error(&dir);
    assert!(
        refusal.contains(QWEN35_DENSE_NO_CUDA),
        "the default cuda answer for the qwen3.8-27B shape must stay the qwen3.5-dense refusal \
         that points at the wgpu serving path, not a flat-Qwen3Config parse death: {refusal}"
    );
    std::env::set_var(DENSE_OPT_IN_ENV, "1");
    let past_detect_family = load_error(&dir);
    std::env::remove_var(DENSE_OPT_IN_ENV);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !past_detect_family.contains(QWEN35_DENSE_NO_CUDA)
            && !past_detect_family.contains("could not detect model family"),
        "with {DENSE_OPT_IN_ENV}=1 detect_family must stop refusing the 27B shape and the load \
         must die later, at the device or the stub weights: {past_detect_family}"
    );
}

#[test]
fn the_flagship_shape_clears_detect_family_the_family_value_pin_lives_in_the_lib_suite() {
    let dir = stub_checkpoint_dir("flagship", &qwen38_flagship_config_json());
    let err = load_error(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !err.contains(QWEN35_DENSE_NO_CUDA)
            && !err.contains("could not detect model family")
            && !err.contains("deserialize qwen3 config"),
        "Qwen3_5MoeForCausalLM + qwen3_5_moe_text must clear family detection with neither the \
         dense refusal nor the flat-Qwen3Config death; that it lands exactly on \
         ModelFamily::Qwen3_5Moe is pinned by the chat_engine build lib suite \
         (qwen3_8_routing_pins_mirror_release_keys_no_shipped_config_json_was_diffed): {err}"
    );
}

#[test]
fn the_27b_config_parses_on_the_dense_arm_and_is_rejected_by_the_moe_arm() {
    let dense = qwen38_27b_config_json();
    nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(&dense).expect(
        "the wgpu dense decoder and the cuda opt-in arm parse the qwen3.8-27B shape with \
         Qwen3_5DenseConfig",
    );
    nv_models::qwen3_5_moe::Qwen3MoeConfig::from_hf_json_str(&dense).expect_err(
        "the MoE parser must reject the dense 27B so the opt-in arm cannot build experts out of \
         a dense checkpoint",
    );
    nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(&qwen38_flagship_config_json())
        .expect_err("the dense parser must reject the flagship's 512-expert flat config");
}
