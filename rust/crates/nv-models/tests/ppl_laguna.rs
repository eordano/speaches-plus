#![cfg(feature = "cuda")]

mod ppl_common;

use candle_core::{Device, Tensor};
use nv_models::laguna::{Laguna, LagunaConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control,
    first_n_corpus_tokens_after_tokenization,
    print_machine_line_with_acc_and_assert_real_beats_shuffled,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
    TeacherForcedNllFp32SoftmaxF64Sum,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;
mod common;
use common::laguna_xs21_nvfp4_snapshot_dir_env_override_then_home_hub;

const CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_MATCHING_THE_GEMMA4_CHAT_PPL_SHAPE:
    usize = 256;
const CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS: usize = 512;

const LAGUNA_TEMPLATE_DEFAULT_SYSTEM_VERBATIM_FROM_CHAT_TEMPLATE_JINJA: &str =
    "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be \
     helpful to users through natural language conversations.";

fn chat_wrapped_continuation_ids_and_score_start_using_the_checkpoints_laguna_template(
    tok: &Tokenizer,
    corpus: &str,
) -> (Vec<u32>, usize) {
    let ctx = CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_MATCHING_THE_GEMMA4_CHAT_PPL_SHAPE;
    let n = ctx + CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS;
    let all = first_n_corpus_tokens_after_tokenization(tok, corpus, n);
    let ctx_text = tok.decode(&all[..ctx], false).expect("decode context slice");
    let prompt = format!(
        "\u{3008}|EOS|\u{3009}<system>{LAGUNA_TEMPLATE_DEFAULT_SYSTEM_VERBATIM_FROM_CHAT_TEMPLATE_JINJA}</system>\n<user>Continue the following text, staying in the same style, with no commentary:\n\n{ctx_text}</user>\n<assistant></think>"
    );
    let prompt_enc = tok
        .encode(prompt.as_str(), false)
        .expect("tokenize chat prompt");
    let mut ids: Vec<u32> = prompt_enc.get_ids().to_vec();
    let score_start = ids.len();
    ids.extend_from_slice(&all[ctx..]);
    (ids, score_start)
}

fn score_tail_via_serving_decode_stepping(
    model: &Laguna,
    device: &Device,
    ids: &[u32],
    score_start: usize,
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    let len = ids.len();
    let mut cache = model.new_kv_cache(len + 8).expect("kv cache");
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut skipped_beyond_logit_width = 0usize;
    for p in 0..len - 1 {
        let tokens =
            Tensor::from_vec(vec![ids[p]], (1usize, 1usize), device).expect("token");
        let positions = Tensor::from_vec(vec![p as i32], 1usize, device).expect("position");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .unwrap_or_else(|err| panic!("decode step {p}: {err:#}"));
        if p + 1 < score_start {
            continue;
        }
        let row: Vec<f32> = logits
            .to_dtype(candle_core::DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("host");
        if ids[p + 1] as usize >= row.len() {
            skipped_beyond_logit_width += 1;
            continue;
        }
        acc.add_position_full_vocab_row(&row, ids[p + 1]);
    }
    if skipped_beyond_logit_width > 0 {
        eprintln!(
            "PPL-SKIPPED {skipped_beyond_logit_width} positions whose true token id exceeds the \
             model's logit width (tokenizer vocab is padded past the lm head); scored positions \
             exclude them on every config identically"
        );
    }
    acc
}

#[test]
#[ignore = "loads the Laguna-XS-2.1 NVFP4 checkpoint; set NV_PPL_TEST=1 and NV_PPL_CORPUS; chat-wrapped with the checkpoint's own template (default Poolside system, enable_thinking=false renders <assistant></think>), scored on the cuda decode-stepping path"]
fn laguna_chat_wrapped_continuation_teacher_forced_ppl_via_serving_decode() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = laguna_xs21_nvfp4_snapshot_dir_env_override_then_home_hub();
    assert!(
        dir.join("model.safetensors").is_file()
            || dir.join("model.safetensors.index.json").is_file(),
        "checkpoint {dir:?} has no weights"
    );
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let (ids, score_start) =
        chat_wrapped_continuation_ids_and_score_start_using_the_checkpoints_laguna_template(
            &tok, &corpus,
        );

    let device = Device::new_cuda(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = LagunaConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Laguna::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    let label = checkpoint_label_from_snapshot_dir(&dir);
    eprintln!(
        "PPL basis: {label} chat_wrapped prompt_tokens={score_start} scored_tokens={} backend=cuda batch=1 path=decode_stepping target_only_no_dflash_drafter",
        ids.len() - score_start
    );

    let real = score_tail_via_serving_decode_stepping(&model, &device, &ids, score_start);

    let mut shuffled_ids = ids.clone();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled_ids[score_start..],
    );
    let shuffled =
        score_tail_via_serving_decode_stepping(&model, &device, &shuffled_ids, score_start);

    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        "laguna-chat",
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}
