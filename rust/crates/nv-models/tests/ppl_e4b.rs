#![cfg(feature = "wgpu")]

mod common;
use common::CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE;
mod ppl_common;

use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control,
    print_machine_line_with_acc_and_assert_real_beats_shuffled,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
    TeacherForcedNllFp32SoftmaxF64Sum,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const E4B_HOME_HUB_REPOS_W4A16_PACK_QUANTIZED_FIRST_BECAUSE_THE_SPEED_DATA_IS_FROM_THAT_CHECKPOINT:
    [&str; 2] = [
    "models--google--gemma-4-E4B-it-qat-w4a16-ct",
    "models--google--gemma-4-E4B-it",
];

fn e4b_snapshot_dir_env_override_then_home_hub_because_devenv_pins_hf_hub_cache_to_a_fixture_cache(
) -> PathBuf {
    if let Ok(d) = std::env::var("NV_E4B_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    for repo in E4B_HOME_HUB_REPOS_W4A16_PACK_QUANTIZED_FIRST_BECAUSE_THE_SPEED_DATA_IS_FROM_THAT_CHECKPOINT
    {
        let base = PathBuf::from(&home)
            .join(".cache/huggingface/hub")
            .join(repo)
            .join("snapshots");
        let Ok(rd) = std::fs::read_dir(&base) else {
            continue;
        };
        let mut candidates: Vec<PathBuf> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
            .collect();
        candidates.sort();
        if let Some(dir) = candidates.into_iter().next() {
            return dir;
        }
    }
    panic!(
        "no complete E4B snapshot under HOME hub (tried {:?}); set NV_E4B_SNAPSHOT",
        E4B_HOME_HUB_REPOS_W4A16_PACK_QUANTIZED_FIRST_BECAUSE_THE_SPEED_DATA_IS_FROM_THAT_CHECKPOINT
    )
}

const CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS: usize = 512;

fn chat_wrapped_continuation_ids_and_copy_start_rendered_through_the_e4b_snapshot_jinja(
    dir: &std::path::Path,
    tok: &Tokenizer,
    corpus: &str,
    bos: u32,
) -> (Vec<u32>, usize) {
    let (ids, score_start) =
        ppl_common::chat_wrapped_continuation_ids_and_copy_start_rendered_through_the_snapshot_jinja(
            dir,
            tok,
            corpus,
            bos,
            CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE,
            CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS,
        );
    let sot_marker = tok
        .token_to_id("<|turn>")
        .expect("E4B <|turn> must be a single special token in tokenizer.json");
    let eot_marker = tok
        .token_to_id("<turn|>")
        .expect("E4B <turn|> must be a single special token in tokenizer.json");
    assert!(
        ids[..score_start].contains(&sot_marker) && ids[..score_start].contains(&eot_marker),
        "E4B turn markers must encode to their special ids, not be split into text pieces"
    );
    (ids, score_start)
}

fn score_tail_via_wgpu_serving_decode_step_logits_full_vocab(
    m: &mut Gemma4E4bWgpu,
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
#[ignore = "loads a real MatFormer E4B checkpoint (W4A16 pack-quantized when cached, dense bf16 fallback) onto the wgpu backend; set NV_PPL_TEST=1 and NV_PPL_CORPUS -- gives the family with only speed data a comparable quality axis"]
fn e4b_chat_wrapped_continuation_teacher_forced_ppl_via_wgpu_serving_decode() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir =
        e4b_snapshot_dir_env_override_then_home_hub_because_devenv_pins_hf_hub_cache_to_a_fixture_cache();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let (ids, score_start) =
        chat_wrapped_continuation_ids_and_copy_start_rendered_through_the_e4b_snapshot_jinja(
            &dir, &tok, &corpus, bos,
        );

    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());

    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
    let mut m = Gemma4E4bWgpu::from_loader(config, &loader, ids.len() + 8).expect("wgpu model");
    drop(loader);

    let label = checkpoint_label_from_snapshot_dir(&dir);
    eprintln!(
        "PPL basis: {label} chat_wrapped prompt_tokens={score_start} scored_tokens={} backend=wgpu batch=1 path=decode_step_logits passes_per_token={} weight_bytes_per_token={}",
        ids.len() - score_start,
        m.pass_count(),
        m.weight_bytes_per_token()
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
        "e4b-chat-wgpu",
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}
