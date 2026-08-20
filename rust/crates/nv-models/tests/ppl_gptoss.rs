#![cfg(feature = "wgpu")]

mod common;
use common::CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE;
mod ppl_common;

use nv_models::gpt_oss_wgpu::{GptOssConfig, GptOssWgpu};
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control,
    first_n_corpus_tokens_after_tokenization, print_machine_line_with_acc_and_assert_real_beats_shuffled,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
    TeacherForcedNllFp32SoftmaxF64Sum,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn gptoss_mxfp4_snapshot_dir_env_override_then_home_hub_because_devenv_pins_hf_hub_cache_to_a_fixture_cache(
) -> PathBuf {
    if let Ok(d) = std::env::var("NV_GPTOSS_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base =
        PathBuf::from(&home).join(".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("gpt-oss-20b snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no complete gpt-oss-20b MXFP4 snapshot under HOME hub; set NV_GPTOSS_SNAPSHOT")
}

const CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS: usize = 512;

const HARMONY_SYSTEM_CURRENT_DATE_PINNED_SO_EVERY_RUN_SCORES_THE_SAME_PROMPT: &str = "2026-08-15";

fn harmony_chat_wrapped_continuation_ids_and_copy_start_no_bos_because_the_template_emits_none(
    tok: &Tokenizer,
    corpus: &str,
) -> (Vec<u32>, usize) {
    let ctx = CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE;
    let n = ctx + CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS;
    let all = first_n_corpus_tokens_after_tokenization(tok, corpus, n);
    let ctx_text = tok.decode(&all[..ctx], false).expect("decode context slice");
    let date = HARMONY_SYSTEM_CURRENT_DATE_PINNED_SO_EVERY_RUN_SCORES_THE_SAME_PROMPT;
    let prompt = format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: {date}\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>Continue the following text, staying in the same style, with no commentary:\n\n{ctx_text}<|end|><|start|>assistant<|channel|>final<|message|>"
    );
    let start_marker = tok
        .token_to_id("<|start|>")
        .expect("harmony <|start|> must be a single special token in tokenizer.json");
    let channel_marker = tok
        .token_to_id("<|channel|>")
        .expect("harmony <|channel|> must be a single special token in tokenizer.json");
    let prompt_enc = tok.encode(prompt.as_str(), false).expect("tokenize chat prompt");
    let mut ids: Vec<u32> = prompt_enc.get_ids().to_vec();
    assert!(
        ids.contains(&start_marker) && ids.contains(&channel_marker),
        "harmony turn markers must encode to their special ids, not be split into text pieces"
    );
    let score_start = ids.len();
    ids.extend_from_slice(&all[ctx..]);
    (ids, score_start)
}

fn score_tail_via_wgpu_serving_decode_step_logits_full_vocab(
    m: &mut GptOssWgpu,
    ids: &[u32],
    score_start: usize,
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    assert_eq!(
        m.current_pos(),
        0,
        "wgpu model must start each arm at position 0; call reset() between arms"
    );
    let vocab = m.config().vocab_size;
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    for p in 0..ids.len() - 1 {
        let (_argmax_token, row) = m
            .decode_step_logits(ids[p])
            .unwrap_or_else(|err| panic!("wgpu decode step {p}: {err:#}"));
        if p + 1 < score_start {
            continue;
        }
        assert_eq!(
            row.len(),
            vocab,
            "decode_step_logits must return the full vocab row, not a top-k slice"
        );
        acc.add_position_full_vocab_row(&row, ids[p + 1]);
    }
    acc
}

#[test]
#[ignore = "loads the ~13 GB gpt-oss-20b MXFP4 checkpoint onto the wgpu backend; set NV_PPL_TEST=1 and NV_PPL_CORPUS -- gives the family with only speed data (gptoss_wgpu_real_weights_decode) a comparable quality axis"]
fn gptoss_chat_wrapped_continuation_teacher_forced_ppl_via_wgpu_serving_decode() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir =
        gptoss_mxfp4_snapshot_dir_env_override_then_home_hub_because_devenv_pins_hf_hub_cache_to_a_fixture_cache();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let (ids, score_start) =
        harmony_chat_wrapped_continuation_ids_and_copy_start_no_bos_because_the_template_emits_none(
            &tok, &corpus,
        );

    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());

    let config = GptOssConfig::from_hf_json_file(dir.join("config.json")).expect("config.json");
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
    let mut m = GptOssWgpu::from_loader(config, &loader, ids.len() + 8).expect("wgpu model");
    drop(loader);

    let label = checkpoint_label_from_snapshot_dir(&dir);
    eprintln!(
        "PPL basis: {label} chat_wrapped prompt_tokens={score_start} scored_tokens={} backend=wgpu batch=1 path=decode_step_logits passes_per_token={}",
        ids.len() - score_start,
        m.pass_count()
    );

    let real = score_tail_via_wgpu_serving_decode_step_logits_full_vocab(&mut m, &ids, score_start);

    let mut shuffled_ids = ids.clone();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled_ids[score_start..],
    );
    m.reset().expect("reset between arms");
    let shuffled =
        score_tail_via_wgpu_serving_decode_step_logits_full_vocab(&mut m, &shuffled_ids, score_start);

    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        "gptoss-chat-wgpu",
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}
