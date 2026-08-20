#![cfg(feature = "cuda")]

#[path = "../crates/nv-models/tests/ppl_common/mod.rs"]
#[allow(dead_code)]
mod ppl_common;

#[path = "common/qwen38_fixture.rs"]
#[allow(dead_code)]
mod qwen38_fixture;

use candle_core::{Device, Tensor};
use nv_models::qwen3_5_moe::Qwen3Moe;
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
use qwen38_fixture::{
    host_row_f32, load_qwen38_dense_on_the_cuda_serving_arm,
    qwen38_nvfp4_snapshot_dir_env_override_then_home_hub,
};
use speaches_plus::oapi::chat_template::ChatTemplate;
use tokenizers::Tokenizer;

const CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_MATCHING_THE_GEMMA4_CHAT_SCORER: usize =
    256;
const CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS: usize = 512;

static ONE_QWEN38_ON_THE_CARD_AT_A_TIME_BECAUSE_LIBTEST_PARALLELISM_CO_LOADS_TWO_22GB_MODELS_AND_OOMS:
    std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialize_this_gpu_test() -> std::sync::MutexGuard<'static, ()> {
    ONE_QWEN38_ON_THE_CARD_AT_A_TIME_BECAUSE_LIBTEST_PARALLELISM_CO_LOADS_TWO_22GB_MODELS_AND_OOMS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

pub fn score_tail_via_serving_decode_stepping(
    model: &Qwen3Moe,
    device: &Device,
    ids: &[u32],
    score_start: usize,
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    let len = ids.len();
    let mut cache = model.new_kv_cache(len + 8).expect("kv cache");
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut skipped_beyond_logit_width = 0usize;
    for p in 0..len - 1 {
        let tokens = Tensor::from_vec(vec![ids[p]], (1usize, 1usize), device).expect("token");
        let positions = Tensor::from_vec(vec![p as i32], 1usize, device).expect("position");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .unwrap_or_else(|err| panic!("decode step {p}: {err:#}"));
        if p + 1 < score_start {
            continue;
        }
        let row = host_row_f32(&logits);
        if ids[p + 1] as usize >= row.len() {
            skipped_beyond_logit_width += 1;
            continue;
        }
        acc.add_position_full_vocab_row(&row, ids[p + 1]);
    }
    assert_eq!(
        skipped_beyond_logit_width, 0,
        "qwen3.8 declares vocab_size 248320 on both the tokenizer and the lm head, so a target \
         beyond the logit width is a wiring bug rather than tokenizer padding"
    );
    acc
}

fn chat_wrapped_continuation_ids_and_score_start_via_the_real_template_thinking_off(
    template: &ChatTemplate,
    tok: &Tokenizer,
    corpus: &str,
) -> (Vec<u32>, usize) {
    let ctx = CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_MATCHING_THE_GEMMA4_CHAT_SCORER;
    let n = ctx + CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS;
    let all = first_n_corpus_tokens_after_tokenization(tok, corpus, n);
    let ctx_text = tok.decode(&all[..ctx], false).expect("decode context slice");
    let msgs = serde_json::json!([{
        "role": "user",
        "content": format!(
            "Continue the following text, staying in the same style, with no commentary:\n\n{ctx_text}"
        )
    }]);
    let mut think_off = std::collections::BTreeMap::new();
    think_off.insert("enable_thinking".to_string(), serde_json::json!(false));
    let prompt = template
        .render_with_kwargs(&msgs, None, true, &think_off)
        .expect("render real qwen3.8 template");
    assert!(
        prompt.ends_with("<think>\n\n</think>\n\n"),
        "a teacher-forced verbatim continuation is scored on the template's own thinking-off \
         arm, whose generation prompt pre-closes the thought block so no scored token sits \
         inside <think>: {prompt:?}"
    );
    let prompt_enc = tok
        .encode(prompt.as_str(), false)
        .expect("tokenize chat prompt");
    let mut ids: Vec<u32> = prompt_enc.get_ids().to_vec();
    let score_start = ids.len();
    ids.extend_from_slice(&all[ctx..]);
    (ids, score_start)
}

#[test]
#[ignore = "loads the ~16 GB Qwen3.8-27B NVFP4 checkpoint; set NV_PPL_TEST=1 and NV_PPL_CORPUS -- chat-wrapped via the checkpoint's own chat_template.jinja, scored on the eager serving decode path (the same forward_with_cache the cuda dense serve arm runs)"]
fn qwen38_chat_wrapped_continuation_teacher_forced_ppl_via_serving_decode() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let _solo = serialize_this_gpu_test();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = qwen38_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let template = ChatTemplate::load(&dir).expect(
        "the qwen3.8 snapshot ships chat_template.jinja and it must compile under minijinja; a \
         hand-built prompt here would be the #95-class artifact this suite exists to prevent",
    );
    let (ids, score_start) =
        chat_wrapped_continuation_ids_and_score_start_via_the_real_template_thinking_off(
            &template, &tok, &corpus,
        );
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let model = load_qwen38_dense_on_the_cuda_serving_arm(&dir, &device);
    let label = checkpoint_label_from_snapshot_dir(&dir);
    eprintln!(
        "PPL basis: {label} chat_wrapped=real_template_thinking_off prompt_tokens={score_start} \
         scored_tokens={} backend=cuda batch=1 path=eager_decode_stepping",
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
        "qwen38-chat",
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}

fn score_corpus_in_512_blocks_one_prefill_per_fresh_cache(
    model: &Qwen3Moe,
    device: &Device,
    corpus_tokens: &[u32],
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    let block = ppl_block_len_with_debug_override_which_makes_the_run_noncanonical();
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
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
        let flat = host_row_f32(&logits);
        assert_eq!(flat.len() % k, 0, "logit rows not divisible by block len");
        let vocab = flat.len() / k;
        let scored_before = acc.scored_positions;
        let nll_before = acc.sum_neg_ln_p_f64;
        for i in 0..k - 1 {
            assert!(
                (ids[i + 1] as usize) < vocab,
                "target id {} exceeds logit width {vocab}; qwen3.8 pads nothing past its lm head",
                ids[i + 1]
            );
            acc.add_position_full_vocab_row(&flat[i * vocab..(i + 1) * vocab], ids[i + 1]);
        }
        eprintln!(
            "PPL-BLOCK start={bstart} scored={} mean_nll={:.3}",
            acc.scored_positions - scored_before,
            (acc.sum_neg_ln_p_f64 - nll_before) / (acc.scored_positions - scored_before) as f64
        );
        bstart = bend;
    }
    acc
}

#[test]
#[ignore = "loads the ~16 GB Qwen3.8-27B NVFP4 checkpoint; set NV_PPL_TEST=1 and NV_PPL_CORPUS -- raw-corpus PPL is a valid eval for the qwen family (no bos prefix, 512 blocks, fresh cache)"]
fn qwen38_raw_corpus_teacher_forced_ppl_512_blocks_fresh_cache_no_cross_block_context() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let _solo = serialize_this_gpu_test();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = qwen38_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let n = PPL_CORPUS_SLICE_TOKENS_N_2048_CHOSEN_SO_31B_LOAD_PLUS_FOUR_EAGER_512_BLOCKS_FITS_2_TO_3_MIN;
    let slice = first_n_corpus_tokens_after_tokenization(&tok, &corpus, n);

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let model = load_qwen38_dense_on_the_cuda_serving_arm(&dir, &device);
    let label = checkpoint_label_from_snapshot_dir(&dir);
    eprintln!("PPL basis: {label} corpus_slice_tokens={n} block=512 backend=cuda batch=1");

    let real = score_corpus_in_512_blocks_one_prefill_per_fresh_cache(&model, &device, &slice);

    let mut shuffled_slice = slice.clone();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled_slice,
    );
    let shuffled =
        score_corpus_in_512_blocks_one_prefill_per_fresh_cache(&model, &device, &shuffled_slice);

    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        "qwen38-raw",
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}
