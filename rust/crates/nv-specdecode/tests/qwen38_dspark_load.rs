use nv_specdecode::dflash::{
    adaptive_submit_len, DFlashSpeculatorConfig, DSparkHeads, LoadedDFlashDrafter,
};

const DSPARK_CONFIG_JSON: &str = r#"{
  "architectures": ["DSparkDraftModel"],
  "block_size": 7,
  "confidence_head_with_markov": true,
  "dflash_config": {
    "attention_mode": "gqa",
    "confidence_head_alpha": 1.0,
    "confidence_head_with_markov": true,
    "enable_confidence_head": true,
    "markov_head_type": "vanilla",
    "markov_rank": 256,
    "mask_token_id": 248077,
    "projector_type": "dspark",
    "target_layer_ids": [4, 16, 28, 40, 52]
  },
  "enable_confidence_head": true,
  "head_dim": 128,
  "hidden_size": 5120,
  "intermediate_size": 10240,
  "layer_types": ["full_attention","full_attention","full_attention","full_attention","full_attention"],
  "markov_head_type": "vanilla",
  "markov_rank": 256,
  "max_position_embeddings": 262144,
  "model_type": "qwen3",
  "num_attention_heads": 40,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 64,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "factor": 32.0,
    "original_max_position_embeddings": 8192,
    "rope_theta": 10000000,
    "rope_type": "yarn"
  },
  "tie_word_embeddings": false,
  "vocab_size": 248320
}"#;

#[test]
fn dspark_config_parses_yarn_and_markov_rank_and_the_block_size_convention_differs_by_one_row() {
    let cfg = DFlashSpeculatorConfig::from_hf_json_str(DSPARK_CONFIG_JSON).unwrap();
    assert_eq!(cfg.block_size, 7, "DSpark ships block_size 7");
    assert_eq!(
        cfg.query_rows(),
        8,
        "our query_rows is 1 + block_size, so a DSpark block_size of 7 gives us EIGHT noise rows \
         while dflash.py::spec_generate builds only block_size of them ([anchor, MASK x 6]) and \
         reads block_size draft logits: our row 8 is a position the checkpoint never saw in \
         training, so a like-for-like DSpark block needs block_size 6 in this config"
    );
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.target_vocab_size, 248320);
    assert_eq!(cfg.draft_vocab_size, 248320, "no vocab compression, so d2t/t2d stay optional");
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.num_attention_heads, 40);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.mask_token_id, 248077);
    assert_eq!(cfg.aux_hidden_state_layer_ids, vec![4, 16, 28, 40, 52]);
    assert_eq!(cfg.fc_in_dim(), 25600, "fc.weight is [5120, 25600]");
    assert_eq!(cfg.markov_rank, 256);
    assert!(!cfg.tied_embeddings);
    assert!(
        !cfg.any_masked_layer(),
        "all five layers are full_attention and the reference calls the drafter with \
         is_causal=False, so the block is non-causal"
    );
    let yarn = cfg.rope_yarn.expect("rope_type yarn must be parsed, not silently dropped");
    assert_eq!(yarn.factor, 32.0);
    assert_eq!(yarn.original_max_position_embeddings, 8192);
    assert!(
        (yarn.attention_factor() - (0.1f64 * 32.0f64.ln() + 1.0)).abs() < 1e-12,
        "YaRN mscale is 0.1*ln(factor)+1"
    );
    let inv = yarn.inv_freq(cfg.head_dim, cfg.rope_theta as f64);
    assert_eq!(inv.len(), cfg.head_dim / 2);
    let plain: Vec<f64> = (0..cfg.head_dim / 2)
        .map(|i| 1.0 / (cfg.rope_theta as f64).powf((i as f64 * 2.0) / cfg.head_dim as f64))
        .collect();
    assert!(
        (inv[0] - plain[0]).abs() < 1e-12,
        "the highest-frequency dim stays pure extrapolation under YaRN"
    );
    let last = inv.len() - 1;
    assert!(
        inv[last] < plain[last] * 0.9,
        "the lowest-frequency dims must be interpolated down by the factor, got {} vs plain {}",
        inv[last],
        plain[last]
    );
}

#[test]
fn adaptive_submit_len_truncates_on_the_cumulative_predicted_accept_probability() {
    assert_eq!(
        adaptive_submit_len(&[0.9, 0.9, 0.9, 0.9], 0.7),
        3,
        "0.9^3 = 0.729 clears 0.7 and 0.9^4 = 0.6561 does not"
    );
    assert_eq!(adaptive_submit_len(&[0.9, 0.9, 0.9, 0.9], 0.5), 4);
    assert_eq!(adaptive_submit_len(&[0.99; 8], 0.5), 8);
    assert_eq!(
        adaptive_submit_len(&[0.1, 0.9, 0.9], 0.5),
        1,
        "a step always submits at least one row so decode cannot stall"
    );
}

#[cfg(feature = "cuda")]
mod real_weights {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use std::path::PathBuf;

    fn gated_dirs() -> Option<(PathBuf, PathBuf)> {
        if std::env::var("NV_DSPARK_TEST").ok().as_deref() != Some("1") {
            eprintln!("skip: set NV_DSPARK_TEST=1");
            return None;
        }
        let draft = PathBuf::from(std::env::var("NV_DSPARK_DRAFT_DIR").ok()?);
        let target = PathBuf::from(std::env::var("NV_DSPARK_TARGET_DIR").ok()?);
        if !draft.join("model.safetensors").is_file() {
            eprintln!("skip: no model.safetensors under {}", draft.display());
            return None;
        }
        Some((draft, target))
    }

    #[test]
    #[ignore]
    fn qwen38_dspark_headless_drafter_loads_against_the_target_embed_and_fp8_lm_head() {
        let Some((draft_dir, target_dir)) = gated_dirs() else {
            return;
        };
        let device = Device::new_cuda(0).unwrap();
        let cfg =
            DFlashSpeculatorConfig::from_hf_json_file(&draft_dir.join("config.json")).unwrap();

        let target = nv_weights::WeightLoader::open_dir(&target_dir, &device).unwrap();
        let embed = nv_layers::weight_load::load_tensor(
            &target,
            "model.language_model.embed_tokens.weight",
            &[cfg.target_vocab_size, cfg.hidden_size],
            DType::BF16,
        )
        .unwrap();
        assert!(
            nv_layers::linear::checkpoint_module_is_fp8_e4m3_weight_with_scale(&target, "lm_head"),
            "the qwen3.8 NVFP4 checkpoint ships lm_head as F8_E4M3 plus a per-row scale; a \
             drafter that borrowed embed_tokens instead would be reading a different matrix, \
             because text_config.tie_word_embeddings is false"
        );
        let lm_head = nv_layers::linear::fp8_e4m3_rowscale_checkpoint_dequant_linear(
            &target,
            "lm_head",
            cfg.target_vocab_size,
            cfg.hidden_size,
            DType::BF16,
        )
        .unwrap();

        let drafter = LoadedDFlashDrafter::try_load_with_target_heads(
            &draft_dir,
            &device,
            Some(&embed),
            Some(lm_head),
        )
        .expect("headless DSpark drafter must load once the target supplies embed and lm_head");

        let heads = drafter
            .dspark_heads()
            .expect("both DSpark heads must be consumed, not silently dropped");
        assert_eq!(heads.markov_rank(), 256);

        let m = drafter.config().query_rows();
        let ctx_rows = 4usize;
        let aux = Tensor::randn(
            0f32,
            0.02f32,
            (ctx_rows, drafter.config().fc_in_dim()),
            &device,
        )
        .unwrap()
        .to_dtype(drafter.dtype())
        .unwrap();
        let proj = drafter.project_aux(&aux).unwrap();
        let positions: Vec<u32> = (0..ctx_rows as u32).collect();
        let ctx = drafter.precompute_context_kv(&proj, &positions).unwrap();

        let anchor = 1234u32;
        let plain = drafter.draft_block_all_rows(&ctx, anchor).unwrap();
        assert_eq!(plain.len(), m);

        let dspark = drafter.draft_block_dspark(&ctx, anchor).unwrap();
        assert_eq!(dspark.ids.len(), m);
        assert_eq!(dspark.accept_probs.len(), m);
        assert_eq!(
            dspark.submit_len, m,
            "NV_DSPARK_ADAPTIVE is unset here, so the block stays fixed-k"
        );
        for (row, id) in dspark.ids.iter().enumerate() {
            assert!(
                (*id as usize) < cfg.target_vocab_size,
                "row {row} drafted {id}, outside the 248320 vocab"
            );
        }
        for (row, p) in dspark.accept_probs.iter().enumerate() {
            assert!(
                p.is_finite() && *p > 0.0 && *p < 1.0,
                "row {row} accept probability {p} is not a probability"
            );
        }
        assert!(
            dspark.ids != plain,
            "the markov head must change at least one drafted row; identical output would mean \
             the rank-256 bigram bias was loaded and then never added"
        );
        eprintln!(
            "dspark_load ok: rows={m} accept_probs={:?} ids={:?}",
            dspark.accept_probs, dspark.ids
        );
    }

    #[test]
    #[ignore]
    fn qwen38_dspark_drafter_refuses_to_load_without_a_target_lm_head() {
        let Some((draft_dir, _)) = gated_dirs() else {
            return;
        };
        let device = Device::new_cuda(0).unwrap();
        let msg = match LoadedDFlashDrafter::try_load(&draft_dir, &device) {
            Ok(_) => panic!("a headless checkpoint must not load silently against no heads at all"),
            Err(e) => format!("{e:#}"),
        };
        assert!(
            msg.contains("headless") || msg.contains("HEADLESS") || msg.contains("62 tensors"),
            "the refusal must name the headless contract, got: {msg}"
        );
    }

    #[test]
    #[ignore]
    fn dspark_head_tensor_names_are_the_four_the_checkpoint_ships() {
        assert_eq!(DSparkHeads::MARKOV_W1, "markov_head.markov_w1.weight");
        assert_eq!(DSparkHeads::MARKOV_W2, "markov_head.markov_w2.weight");
        assert_eq!(DSparkHeads::CONFIDENCE_W, "confidence_head.proj.weight");
        assert_eq!(DSparkHeads::CONFIDENCE_B, "confidence_head.proj.bias");
    }
}
