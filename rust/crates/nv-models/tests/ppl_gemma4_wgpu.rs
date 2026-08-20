#![cfg(feature = "wgpu")]

mod ppl_common;

use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_wgpu::{host_weights_from_loader, weight_format_boot_line, Gemma4Wgpu};
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control,
    print_machine_line_with_acc_and_assert_real_beats_shuffled,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
    TeacherForcedNllFp32SoftmaxF64Sum,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;
mod common;
use common::gemma4_nvfp4_snapshot_dir_env_override_then_home_hub;

const CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_A_31B_RUN_FITS_2_MIN: usize = 256;
const CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS: usize = 512;

fn chat_wrapped_continuation_ids_and_copy_start(
    dir: &std::path::Path,
    tok: &Tokenizer,
    corpus: &str,
    bos: u32,
) -> (Vec<u32>, usize) {
    ppl_common::chat_wrapped_continuation_ids_and_copy_start_rendered_through_the_snapshot_jinja(
        dir,
        tok,
        corpus,
        bos,
        CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_A_31B_RUN_FITS_2_MIN,
        CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS,
    )
}

fn score_tail_via_wgpu_serving_decode_step_logits_full_vocab(
    m: &mut Gemma4Wgpu,
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

fn gemma4_moe_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_GEMMA4_MOE_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("gemma4-moe snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no complete gemma4-moe snapshot under HOME hub; set NV_GEMMA4_MOE_DIR")
}

fn score_tail_via_moe_wgpu_serving_decode_step_logits_full_vocab(
    m: &mut nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu,
    ids: &[u32],
    score_start: usize,
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    assert_eq!(
        m.current_pos(),
        0,
        "wgpu model must start each arm at position 0; call reset() between arms"
    );
    let vocab = m.config().base.vocab_size;
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    for p in 0..ids.len() - 1 {
        let (_argmax_token, row) = m
            .decode_step_logits(ids[p])
            .unwrap_or_else(|err| panic!("wgpu moe decode step {p}: {err:#}"));
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
#[ignore = "loads the gemma-4-26B-A4B MoE (bf16 checkpoint, on-load int8-dense/w4-expert) onto the wgpu backend; set NV_PPL_TEST=1 and NV_PPL_CORPUS -- chat-wrapped like the dense arm so the MoE gets a comparable quality axis"]
fn gemma4_moe_chat_wrapped_continuation_teacher_forced_ppl_via_wgpu_serving_decode() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = gemma4_moe_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let (ids, score_start) =
        chat_wrapped_continuation_ids_and_copy_start(&dir, &tok, &corpus, bos);

    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());

    let cfg = nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
    let mut m = nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu::from_loader(cfg, &loader, ids.len() + 8)
        .expect("wgpu moe model");
    drop(loader);

    let label = checkpoint_label_from_snapshot_dir(&dir);
    eprintln!(
        "PPL basis: {label} chat_wrapped prompt_tokens={score_start} scored_tokens={} backend=wgpu batch=1 path=decode_step_logits quant=on-load-int8-dense-w4-expert",
        ids.len() - score_start
    );

    let real = score_tail_via_moe_wgpu_serving_decode_step_logits_full_vocab(&mut m, &ids, score_start);

    let mut shuffled_ids = ids.clone();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled_ids[score_start..],
    );
    m.reset().expect("reset between arms");
    let shuffled = score_tail_via_moe_wgpu_serving_decode_step_logits_full_vocab(
        &mut m,
        &shuffled_ids,
        score_start,
    );

    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        "gemma4-moe-chat-wgpu",
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}

#[test]
#[ignore = "loads the ~18 GB Gemma-4-31B NVFP4 checkpoint onto the wgpu backend; set NV_PPL_TEST=1 and NV_PPL_CORPUS -- chat-wrapped like the cuda twin so NV_G4_WGPU_W8_FFN configs get a comparable quality axis"]
fn gemma4_chat_wrapped_continuation_teacher_forced_ppl_via_wgpu_serving_decode() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = gemma4_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let (ids, score_start) =
        chat_wrapped_continuation_ids_and_copy_start(&dir, &tok, &corpus, bos);

    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());

    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
    let host = host_weights_from_loader(&config, &loader).expect("host weight staging");
    drop(loader);
    let mut m = Gemma4Wgpu::new(config, &host, ids.len() + 8).expect("wgpu model");
    drop(host);

    let label = checkpoint_label_from_snapshot_dir(&dir);
    eprintln!(
        "PPL basis: {label} chat_wrapped prompt_tokens={score_start} scored_tokens={} backend=wgpu batch=1 path=decode_step_logits w8_ffn_env={:?} formats=[{}]",
        ids.len() - score_start,
        std::env::var("NV_G4_WGPU_W8_FFN").ok(),
        weight_format_boot_line()
    );

    let real = score_tail_via_wgpu_serving_decode_step_logits_full_vocab(&mut m, &ids, score_start);

    let mut shuffled_ids = ids.clone();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled_ids[score_start..],
    );
    m.reset();
    let shuffled =
        score_tail_via_wgpu_serving_decode_step_logits_full_vocab(&mut m, &shuffled_ids, score_start);

    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        "gemma4-chat-wgpu",
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}
