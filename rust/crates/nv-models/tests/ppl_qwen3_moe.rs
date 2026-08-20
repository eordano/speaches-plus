#![cfg(feature = "cuda")]

mod ppl_common;

use candle_core::{Device, Tensor};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control,
    first_n_corpus_tokens_after_tokenization,
    ppl_block_len_with_debug_override_which_makes_the_run_noncanonical,
    print_machine_line_with_acc_and_assert_real_beats_shuffled,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
    TeacherForcedNllFp32SoftmaxF64Sum,
    PPL_CORPUS_SLICE_TOKENS_N_2048_CHOSEN_SO_31B_LOAD_PLUS_FOUR_EAGER_512_BLOCKS_FITS_2_TO_3_MIN,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn qwen36_nvfp4_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN36_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("qwen3.6 snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
        .expect("no complete qwen3.6 NVFP4 snapshot under HOME hub; set NV_QWEN36_DIR")
}

fn score_corpus_in_512_blocks_first_token_input_only_one_prefill_per_fresh_cache(
    model: &Qwen3Moe,
    device: &Device,
    corpus_tokens: &[u32],
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    let block = ppl_block_len_with_debug_override_which_makes_the_run_noncanonical();
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut skipped_beyond_logit_width = 0usize;
    let mut bstart = 0usize;
    while bstart < corpus_tokens.len() {
        let bend = (bstart + block).min(corpus_tokens.len());
        let ids = &corpus_tokens[bstart..bend];
        let k = ids.len();
        let mut cache = model.new_kv_cache(k + 8).expect("kv cache");
        let tokens = Tensor::from_vec(ids.to_vec(), (1usize, k), device).expect("tokens");
        let positions =
            Tensor::from_vec((0..k as i32).collect::<Vec<_>>(), k, device).expect("positions");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .unwrap_or_else(|err| panic!("forward_with_cache block {bstart}: {err:#}"));
        let flat: Vec<f32> = logits
            .to_dtype(candle_core::DType::F32)
            .expect("logits f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to host");
        assert_eq!(flat.len() % k, 0, "logit rows not divisible by block len");
        let vocab = flat.len() / k;
        let nll_before = acc.sum_neg_ln_p_f64;
        let scored_before = acc.scored_positions;
        for i in 0..k - 1 {
            if ids[i + 1] as usize >= vocab {
                skipped_beyond_logit_width += 1;
                continue;
            }
            acc.add_position_full_vocab_row(&flat[i * vocab..(i + 1) * vocab], ids[i + 1]);
        }
        eprintln!(
            "PPL-BLOCK start={bstart} scored={} mean_nll={:.3}",
            acc.scored_positions - scored_before,
            (acc.sum_neg_ln_p_f64 - nll_before) / (acc.scored_positions - scored_before) as f64
        );
        bstart = bend;
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
#[ignore = "loads the ~22 GB Qwen3.6-35B NVFP4 checkpoint; set NV_PPL_TEST=1 and NV_PPL_CORPUS"]
fn qwen3_moe_cuda_teacher_forced_ppl_512_blocks_fresh_cache_no_cross_block_context() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = qwen36_nvfp4_snapshot_dir_env_override_then_home_hub();
    assert!(
        dir.join("model.safetensors").is_file()
            || dir.join("model.safetensors.index.json").is_file(),
        "checkpoint {dir:?} has no weights"
    );
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let n = PPL_CORPUS_SLICE_TOKENS_N_2048_CHOSEN_SO_31B_LOAD_PLUS_FOUR_EAGER_512_BLOCKS_FITS_2_TO_3_MIN;
    let slice = first_n_corpus_tokens_after_tokenization(&tok, &corpus, n);

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    let label = checkpoint_label_from_snapshot_dir(&dir);
    eprintln!("PPL basis: {label} corpus_slice_tokens={n} block=512 backend=cuda batch=1");

    let real = score_corpus_in_512_blocks_first_token_input_only_one_prefill_per_fresh_cache(
        &model, &device, &slice,
    );

    let mut shuffled_slice = slice.clone();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled_slice,
    );
    let shuffled = score_corpus_in_512_blocks_first_token_input_only_one_prefill_per_fresh_cache(
        &model,
        &device,
        &shuffled_slice,
    );

    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        "qwen3_moe",
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}
