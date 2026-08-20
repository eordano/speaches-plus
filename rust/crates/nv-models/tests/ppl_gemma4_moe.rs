#![cfg(feature = "wgpu")]

mod common;
use common::CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE;
mod ppl_common;

use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu;
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control,
    print_machine_line_with_acc_and_assert_real_beats_shuffled,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
    TeacherForcedNllFp32SoftmaxF64Sum,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const G4MOE_HOME_HUB_REPO_MATCHING_THE_SWEEP_SPEED_ROW: &str =
    "models--google--gemma-4-26B-A4B-it";

fn g4moe_snapshot_dir_env_override_then_home_hub_because_devenv_pins_hf_hub_cache_to_a_fixture_cache(
) -> PathBuf {
    if let Ok(d) = std::env::var("NV_G4MOE_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(&home)
        .join(".cache/huggingface/hub")
        .join(G4MOE_HOME_HUB_REPO_MATCHING_THE_SWEEP_SPEED_ROW)
        .join("snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("g4moe snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no complete gemma-4-26B-A4B-it snapshot under HOME hub; set NV_G4MOE_SNAPSHOT")
}

const CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS: usize = 512;

fn chat_wrapped_continuation_ids_and_copy_start_rendered_through_the_g4moe_snapshot_jinja(
    dir: &std::path::Path,
    tok: &Tokenizer,
    corpus: &str,
    bos: u32,
) -> (Vec<u32>, usize) {
    for marker in ["<|turn>", "<turn|>", "<|channel>", "<channel|>"] {
        let id = tok.token_to_id(marker).unwrap_or_else(|| {
            panic!("g4moe {marker} must be a single special token in tokenizer.json")
        });
        let enc = tok.encode(marker, false).expect("tokenize marker alone");
        assert_eq!(
            enc.get_ids(),
            &[id],
            "g4moe {marker} must encode to its single special id, not be split into text pieces"
        );
    }
    ppl_common::chat_wrapped_continuation_ids_and_copy_start_rendered_through_the_snapshot_jinja(
        dir,
        tok,
        corpus,
        bos,
        CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE,
        CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS,
    )
}

fn score_tail_via_wgpu_serving_decode_step_logits_full_vocab(
    m: &mut Gemma4MoeWgpu,
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
#[ignore = "loads the real ~26B-A4B MoE checkpoint onto the wgpu backend; set NV_PPL_TEST=1 and NV_PPL_CORPUS -- gives the g4moe-26b-a4b-wgpu sweep row, which has only speed data, a comparable quality axis"]
fn g4moe_chat_wrapped_continuation_teacher_forced_ppl_via_wgpu_serving_decode() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir =
        g4moe_snapshot_dir_env_override_then_home_hub_because_devenv_pins_hf_hub_cache_to_a_fixture_cache();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let (ids, score_start) =
        chat_wrapped_continuation_ids_and_copy_start_rendered_through_the_g4moe_snapshot_jinja(
            &dir, &tok, &corpus, bos,
        );

    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());

    let config = Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
    let mut m = Gemma4MoeWgpu::from_loader(config, &loader, ids.len() + 8).expect("wgpu model");
    drop(loader);

    let label = checkpoint_label_from_snapshot_dir(&dir);
    let flash1_passes_nonzero_proves_decode_step_logits_hits_the_flash_gate = m
        .pass_rows()
        .iter()
        .filter(|(l, _, _, _)| l.starts_with("g4m-at-flash1"))
        .count();
    eprintln!(
        "PPL basis: {label} chat_wrapped prompt_tokens={score_start} scored_tokens={} backend=wgpu batch=1 path=decode_step_logits passes_per_token={} weight_bytes_per_token={} flash1_passes={} kv_fp8={}",
        ids.len() - score_start,
        m.pass_count(),
        m.weight_bytes_per_token(),
        flash1_passes_nonzero_proves_decode_step_logits_hits_the_flash_gate,
        nv_models::gemma4_moe_wgpu::kv_fp8_enabled()
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
        "g4moe-chat-wgpu",
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}
