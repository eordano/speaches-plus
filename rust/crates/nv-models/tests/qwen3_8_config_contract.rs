use nv_models::qwen3::Qwen3Config;
use nv_models::qwen3_5_moe::{LayerType, Qwen3_5DenseConfig, MOE_ONLY_KEYS};
use nv_weights::{QuantScheme, QuantizationConfig};

const QWEN38_27B_CONFIG_JSON: &str = include_str!("qwen3_8_27b_config.json");

const RELEASE_FACT_NUM_HIDDEN_LAYERS_64_NOT_48_AS_BACKEND_SELECT_SELFTEST_LITERAL_ASSUMED: usize =
    64;
const RELEASE_FACT_FULL_ATTENTION_LAYERS_16_EVERY_FOURTH_INDEX_3_7_11_TO_63: usize = 16;

#[test]
fn qwen3_8_27b_config_parses_as_qwen3_5_dense_and_pins_every_release_fact() {
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(QWEN38_27B_CONFIG_JSON)
        .expect("real unsloth/Qwen3.8-27B-NVFP4 config.json must parse as a Qwen3_5 dense config");

    assert_eq!(
        cfg.num_hidden_layers,
        RELEASE_FACT_NUM_HIDDEN_LAYERS_64_NOT_48_AS_BACKEND_SELECT_SELFTEST_LITERAL_ASSUMED
    );
    assert_eq!(cfg.layer_types.len(), cfg.num_hidden_layers);
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.intermediate_size, 17408);
    assert!(cfg.intermediate_size > 0, "dense arm requires a real MLP width");
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.num_attention_heads, 24);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.linear_num_key_heads, 16);
    assert_eq!(cfg.linear_num_value_heads, 48);
    assert_eq!(cfg.linear_key_head_dim, 128);
    assert_eq!(cfg.linear_value_head_dim, 128);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    assert_eq!(cfg.vocab_size, 248320);
    assert_eq!(cfg.max_position_embeddings, 262144);
    assert_eq!(cfg.rope_theta, 10_000_000.0);
    assert_eq!(cfg.partial_rotary_factor, 0.25);
    assert_eq!(cfg.rotary_dim(), 64);
    assert!(cfg.attn_output_gate);
    assert!(!cfg.tie_word_embeddings);
    assert_eq!(cfg.eos_token_id, 248044);
    assert_eq!(cfg.bos_token_id, Some(248044));
    assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);

    let full = cfg
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::FullAttention)
        .count();
    let linear = cfg
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    assert_eq!(
        full,
        RELEASE_FACT_FULL_ATTENTION_LAYERS_16_EVERY_FOURTH_INDEX_3_7_11_TO_63
    );
    assert_eq!(linear, 48);
    for (i, t) in cfg.layer_types.iter().enumerate() {
        let expected = if (i + 1) % 4 == 0 {
            LayerType::FullAttention
        } else {
            LayerType::LinearAttention
        };
        assert_eq!(
            *t, expected,
            "layer {i}: full_attention_interval=4 pins 3x linear then 1x full"
        );
    }
}

#[test]
fn qwen3_8_27b_text_config_carries_no_moe_only_keys() {
    let v: serde_json::Value = serde_json::from_str(QWEN38_27B_CONFIG_JSON).unwrap();
    let text = v.get("text_config").expect("text_config nesting is the contract");
    for key in MOE_ONLY_KEYS {
        assert!(
            text.get(key).is_none(),
            "text_config declares MoE-only key {key}: Qwen3.8-27B is a dense hybrid, not a MoE"
        );
    }
    assert!(v.get("vision_config").is_some(), "vision_config is present (out of scope for text serving)");
}

#[test]
fn legacy_qwen3_config_rejects_qwen3_8_because_fields_nest_under_text_config() {
    let err = Qwen3Config::from_hf_json_str(QWEN38_27B_CONFIG_JSON)
        .expect_err("legacy flat Qwen3Config must reject the text_config-nested Qwen3.8 layout");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("hidden_size") || msg.contains("deserialize") || msg.contains("missing"),
        "rejection should be about the missing top-level fields, got: {msg}"
    );
}

#[test]
fn qwen3_8_27b_quant_config_is_mixed_precision_collapsing_to_fp8_first_group() {
    let q = QuantizationConfig::from_hf_json_str(QWEN38_27B_CONFIG_JSON)
        .expect("compressed-tensors mixed-precision quant config must parse");
    assert_eq!(
        q.scheme,
        QuantScheme::Fp8E4m3,
        "config_groups iterate in BTreeMap key order, so group_0 (fp8: attn q/k/v/o + linear_attn \
         in_proj + lm_head + last-8-layer mlp) is matched before group_1 (nvfp4: mlp gate/up/down); \
         from_hf_value collapses this mixed fp8+nvfp4 checkpoint to a single scalar scheme = fp8"
    );
    assert!(
        q.ignored_modules.iter().any(|m| m == "re:^mtp.*"),
        "quant ignore list must exclude the MTP head"
    );
    assert!(
        q.ignored_modules.iter().any(|m| m.contains("visual")),
        "quant ignore list must exclude the vision tower"
    );
    assert!(
        q.ignored_modules.iter().any(|m| m.contains("linear_attn")),
        "quant ignore list must exclude the gated-deltanet linear_attn projections"
    );
}

#[cfg(feature = "wgpu")]
mod real_weights {
    use nv_models::qwen3_5_dense_wgpu as q3d;
    use nv_models::qwen3_5_moe::Qwen3_5DenseConfig;
    use std::path::PathBuf;

    fn hub_snapshot_dir_multi_root() -> Option<PathBuf> {
        let mut roots: Vec<PathBuf> = Vec::new();
        for key in ["NV_QWEN38_DIR", "NV_MODELS_TEST_HUB", "HF_HUB_CACHE"] {
            if let Ok(p) = std::env::var(key) {
                if !p.is_empty() {
                    roots.push(PathBuf::from(p));
                }
            }
        }
        if let Ok(home) = std::env::var("HOME") {
            roots.push(PathBuf::from(home).join(".cache/huggingface/hub"));
        }
        for root in roots {
            if root.join("config.json").is_file() {
                return Some(root);
            }
            let snaps = root.join("models--unsloth--Qwen3.8-27B-NVFP4/snapshots");
            if let Ok(rd) = std::fs::read_dir(&snaps) {
                if let Some(p) = rd
                    .flatten()
                    .map(|e| e.path())
                    .find(|p| p.join("config.json").is_file())
                {
                    return Some(p);
                }
            }
        }
        None
    }

    #[test]
    #[ignore = "loads the real unsloth/Qwen3.8-27B-NVFP4 checkpoint; set NV_QWEN38_REAL_TEST=1"]
    fn qwen3_8_27b_real_weights_greedy_decode_64_tokens() {
        if std::env::var("NV_QWEN38_REAL_TEST").is_err() {
            eprintln!("[skip] NV_QWEN38_REAL_TEST not set");
            return;
        }
        let dir = hub_snapshot_dir_multi_root()
            .expect("no hydrated unsloth/Qwen3.8-27B-NVFP4 snapshot; set NV_QWEN38_DIR");
        eprintln!("[real] checkpoint={}", dir.display());
        let cfg = Qwen3_5DenseConfig::from_hf_json_file(&dir.join("config.json")).expect("config");

        let loader =
            nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("loader");
        let t0 = std::time::Instant::now();
        let mut model = q3d::Qwen3_5DenseWgpu::from_loader(cfg.clone(), &loader, 512).expect(
            "build Qwen3.8-27B on the qwen3_5 dense wgpu decoder; this path is bf16-tensor-only \
             (loads plain .weight), so the mixed fp8+nvfp4 unsloth checkpoint (weight_packed / \
             weight_scale / weight_global_scale) has no bf16 mlp.*.weight to load: a coherent decode \
             here requires an NVFP4-aware dense loader that does not yet exist",
        );
        eprintln!("[real] loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
        let prompt = "The capital of France is";
        let ids: Vec<u32> = tokenizer
            .encode(prompt, false)
            .expect("encode")
            .get_ids()
            .to_vec();
        let mut last = 0u32;
        for t in &ids {
            last = model.decode_step(*t).expect("prefill step");
        }
        let n_new: usize = 64;
        let mut out_ids: Vec<u32> = Vec::new();
        let t1 = std::time::Instant::now();
        for _ in 0..n_new {
            out_ids.push(last);
            if last == cfg.eos_token_id {
                break;
            }
            last = model.decode_step(last).expect("decode step");
        }
        let per_tok = t1.elapsed().as_secs_f64() * 1000.0 / out_ids.len().max(1) as f64;
        let text = tokenizer.decode(&out_ids, true).unwrap_or_default();
        eprintln!(
            "[real] basis: checkpoint={} backend=wgpu batch=1 new_tokens={} log={}",
            dir.display(),
            out_ids.len(),
            std::env::var("NV_QWEN38_LOG").unwrap_or_else(|_| "<stderr>".into())
        );
        eprintln!("[real] {per_tok:.1} ms/token; continuation={text:?}");
        assert!(
            text.to_lowercase().contains("paris"),
            "greedy continuation of {prompt:?} did not mention Paris: {text:?}"
        );
    }
}
