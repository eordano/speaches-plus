#![cfg(feature = "cuda")]

mod ppl_common;

use candle_core::{Device, Tensor};
use nv_models::qwen3::{Qwen3, Qwen3Config};
use nv_weights::{QuantScheme, QuantizationConfig, WeightLoader};
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    first_n_corpus_tokens_after_tokenization,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
    TeacherForcedNllFp32SoftmaxF64Sum,
    PPL_CORPUS_SLICE_TOKENS_N_2048_CHOSEN_SO_31B_LOAD_PLUS_FOUR_EAGER_512_BLOCKS_FITS_2_TO_3_MIN,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub const QWEN3_WEIGHTS_ARMS_MIRROR_CHAT_ENGINE_BUILD_QWEN3_WEIGHT_QCONFIG: &str =
    "NV_QWEN3_WEIGHTS selects the arm exactly as the serving loader does \
     (chat_engine::build::qwen3_weight_qconfig_default_nvfp4): unset or 'nvfp4' builds \
     Qwen3::from_loader_quantized with QuantScheme::Nvfp4 (weights-only, per-layer q/k/v/o and \
     gate/up/down projections, 16-element ue4m3 block scales; embed, lm_head and norms stay \
     bf16); 'bf16' builds the pre-flip Qwen3::from_loader path";

fn qwen3_dense_snapshot_dir_env_override_then_hub_caches() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN3_DIR") {
        return PathBuf::from(d);
    }
    let mut bases = Vec::new();
    if let Ok(hub) = std::env::var("HF_HUB_CACHE") {
        bases.push(PathBuf::from(hub).join("models--Qwen--Qwen3-Embedding-0.6B/snapshots"));
    }
    if let Ok(home) = std::env::var("HOME") {
        bases.push(
            PathBuf::from(home)
                .join(".cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots"),
        );
    }
    for base in &bases {
        if let Ok(entries) = std::fs::read_dir(base) {
            if let Some(snap) = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
            {
                return snap;
            }
        }
    }
    panic!(
        "no complete Qwen3-Embedding-0.6B snapshot under {bases:?}; set NV_QWEN3_DIR to a plain \
         qwen3 (model_type qwen3, top-level trunk fields) checkpoint dir"
    )
}

fn arm_from_nv_qwen3_weights_env() -> String {
    match std::env::var("NV_QWEN3_WEIGHTS").ok().as_deref() {
        Some("bf16") => "bf16".to_string(),
        None | Some("nvfp4") => "nvfp4".to_string(),
        Some(other) => panic!(
            "NV_QWEN3_WEIGHTS={other} is not an arm; the instrument accepts exactly 'bf16' and \
             'nvfp4' so a typo cannot silently score the wrong configuration"
        ),
    }
}

fn build_qwen3_arm(
    arm: &str,
    cfg: Qwen3Config,
    weights: &WeightLoader,
    device: &Device,
) -> Qwen3 {
    match arm {
        "bf16" => Qwen3::from_loader(cfg, weights, device).expect("bf16 model"),
        _ => {
            let qcfg = QuantizationConfig {
                scheme: QuantScheme::Nvfp4,
                ignored_modules: Vec::new(),
                group_size: None,
            };
            Qwen3::from_loader_quantized(cfg, weights, &qcfg, device).expect("nvfp4 model")
        }
    }
}

fn score_corpus_in_512_blocks_first_token_input_only_one_prefill_per_fresh_cache(
    model: &Qwen3,
    device: &Device,
    corpus_tokens: &[u32],
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    ppl_common::score_corpus_in_blocks_first_token_input_only_one_prefill_per_fresh_cache(
        &mut |ids, bstart| {
            let k = ids.len();
            let mut cache = model.new_kv_cache(k + 8).expect("kv cache");
            let tokens = Tensor::from_vec(ids.to_vec(), (1usize, k), device).expect("tokens");
            let positions =
                Tensor::from_vec((0..k as i32).collect::<Vec<_>>(), k, device).expect("positions");
            model
                .forward(&tokens, &positions, &mut cache)
                .unwrap_or_else(|err| panic!("forward block {bstart}: {err:#}"))
                .to_dtype(candle_core::DType::F32)
                .expect("logits f32")
                .flatten_all()
                .expect("flatten")
                .to_vec1()
                .expect("to host")
        },
        corpus_tokens,
    )
}

#[test]
#[ignore = "loads the real 0.6B; set NV_PPL_TEST=1 and NV_PPL_CORPUS; NV_QWEN3_WEIGHTS=bf16|nvfp4 selects the arm"]
fn qwen3_dense_cuda_teacher_forced_ppl_512_blocks_arm_from_nv_qwen3_weights() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let arm = arm_from_nv_qwen3_weights_env();
    let dir = qwen3_dense_snapshot_dir_env_override_then_hub_caches();
    assert!(
        dir.join("model.safetensors").is_file()
            || dir.join("model.safetensors.index.json").is_file(),
        "checkpoint {dir:?} has no weights"
    );
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let n = PPL_CORPUS_SLICE_TOKENS_N_2048_CHOSEN_SO_31B_LOAD_PLUS_FOUR_EAGER_512_BLOCKS_FITS_2_TO_3_MIN;
    let slice = first_n_corpus_tokens_after_tokenization(&tok, &corpus, n);

    let device = Device::new_cuda(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3Config::from_hf_json_str(&raw_cfg).expect("parse config");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = build_qwen3_arm(&arm, cfg, &weights, &device);
    let label = format!("checkpoint={}", checkpoint_label_from_snapshot_dir(&dir));
    eprintln!("PPL basis: {label} arm={arm} corpus_slice_tokens={n} block=512 backend=cuda batch=1");

    ppl_common::shuffle_tail_score_both_arms_assert_equal_positions_and_print_machine_line(
        &format!("qwen3-{arm}"),
        &label,
        &slice,
        0,
        &mut |arm_ids| {
            score_corpus_in_512_blocks_first_token_input_only_one_prefill_per_fresh_cache(
                &model, &device, arm_ids,
            )
        },
    );
}
