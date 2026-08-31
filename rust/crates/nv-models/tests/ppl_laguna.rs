#![cfg(feature = "cuda")]

mod ctx_timing_common;
mod ppl_common;

use candle_core::{Device, Tensor};
use nv_models::laguna::{Laguna, LagunaConfig};
use nv_models::laguna_step_graph::LagunaStepGraph;
use nv_weights::{QuantizationConfig, WeightLoader};
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control,
    eprintln_ppl_skipped_when_any_true_token_exceeded_the_logit_width,
    first_n_corpus_tokens_after_tokenization,
    print_machine_line_with_acc_and_assert_real_beats_shuffled,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
    TeacherForcedNllFp32SoftmaxF64Sum,
};
use std::sync::Arc;
use tokenizers::Tokenizer;
mod common;
use common::laguna_xs21_nvfp4_snapshot_dir_env_override_then_home_hub;

const CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_MATCHING_THE_GEMMA4_CHAT_PPL_SHAPE:
    usize = 256;
const CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS: usize = 512;

const LAGUNA_TEMPLATE_DEFAULT_SYSTEM_VERBATIM_FROM_CHAT_TEMPLATE_JINJA: &str =
    "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be \
     helpful to users through natural language conversations.";

const NV_LAGUNA_PPL_GRAPH_SELECTS_THE_CAPTURED_STEP_GRAPH_SCORER_BECAUSE_THE_EAGER_SCORER_RUNS_NONE_OF_THE_KERNELS_A_GRAPH_ONLY_ARM_CHANGES_SO_ITS_DIGIT_IDENTICAL_PPL_MEANS_NOT_MEASURED_NOT_HELD:
    &str = "NV_LAGUNA_PPL_GRAPH";

const GREEDY_CHAIN_ANCHOR_STEPS_128_IS_THE_LENGTH_OVER_WHICH_EAGER_AND_STEP_GRAPH_MUST_AGREE_TOKEN_FOR_TOKEN_AND_THAT_AGREEMENT_IS_THE_ONLY_LICENCE_FOR_READING_THE_TWO_SCORERS_ON_ONE_SCALE:
    usize = 128;

fn arm_label_from_the_resolved_value_of_every_numerics_changing_gate_plus_the_shared_fold_prerequisite(
) -> String {
    ctx_timing_common::arm_label_from_the_resolved_value_of_every_gate_accessor_because_a_presence_label_lies_after_a_default_flip(
        &ctx_timing_common::LAGUNA_DECODE_NUMERICS_GATES_EVERY_ONE_APPEARS_RESOLVED_IN_THE_ARM_LABEL,
    )
}

fn score_through_the_captured_step_graph_that_laguna_serve_builds_unconditionally() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call(
        NV_LAGUNA_PPL_GRAPH_SELECTS_THE_CAPTURED_STEP_GRAPH_SCORER_BECAUSE_THE_EAGER_SCORER_RUNS_NONE_OF_THE_KERNELS_A_GRAPH_ONLY_ARM_CHANGES_SO_ITS_DIGIT_IDENTICAL_PPL_MEANS_NOT_MEASURED_NOT_HELD,
    )
}

fn assert_no_graph_only_arm_is_set_without_the_prerequisite_that_lets_it_engage_because_an_inert_arm_scores_digit_identical_and_reads_as_held() {
    nv_config::env_arm_dependency_assert_enabled_arm_requires_prerequisite(
        "NV_LAGUNA_MOE_NORMGATE_FUSE",
        nv_config::env_arm_opt_in_set_to_anything_but_0_reread_every_call(
            "NV_LAGUNA_MOE_NORMGATE_FUSE",
        ),
        "NV_MOE_SHARED_FOLD",
        nv_layers::moe_grouped::folded_shared_enabled(),
        "the fuse stops materialising the normed tensor that the unfolded shared-expert branch consumes, so laguna_graph.rs runs it only with the shared expert folded; without the fold the arm executes nothing and its score is the basis score wearing an arm label",
    )
    .expect("graph-only arm prerequisite");
}

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

struct LagunaChatPplBasis {
    model: Arc<Laguna>,
    device: Device,
    ids: Vec<u32>,
    score_start: usize,
    label: String,
}

fn load_the_checkpoint_and_build_the_chat_wrapped_scoring_sequence() -> LagunaChatPplBasis {
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
    let model = Arc::new(
        Laguna::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model"),
    );
    drop(weights);
    let label = checkpoint_label_from_snapshot_dir(&dir);
    LagunaChatPplBasis {
        model,
        device,
        ids,
        score_start,
        label,
    }
}

fn score_tail_from_one_full_vocab_f32_row_per_position_skipping_true_tokens_past_the_logit_width(
    step_logits_row: &mut dyn FnMut(usize, u32) -> Vec<f32>,
    ids: &[u32],
    score_start: usize,
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    let len = ids.len();
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut skipped_beyond_logit_width = 0usize;
    for p in 0..len - 1 {
        let row = step_logits_row(p, ids[p]);
        if p + 1 < score_start {
            continue;
        }
        if ids[p + 1] as usize >= row.len() {
            skipped_beyond_logit_width += 1;
            continue;
        }
        acc.add_position_full_vocab_row(&row, ids[p + 1]);
    }
    eprintln_ppl_skipped_when_any_true_token_exceeded_the_logit_width(skipped_beyond_logit_width);
    acc
}

fn eager_decode_step_logits_row(
    model: &Laguna,
    device: &Device,
    cache: &mut nv_models::laguna::LagunaKvCache,
    p: usize,
    token: u32,
) -> Vec<f32> {
    let tokens = Tensor::from_vec(vec![token], (1usize, 1usize), device).expect("token");
    let positions = Tensor::from_vec(vec![p as i32], 1usize, device).expect("position");
    let logits = model
        .forward_with_cache(&tokens, &positions, cache)
        .unwrap_or_else(|err| panic!("eager decode step {p}: {err:#}"));
    logits
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("host")
}

fn step_graph_decode_step_logits_row(graph: &mut LagunaStepGraph, p: usize, token: u32) -> Vec<f32> {
    graph
        .step(token)
        .unwrap_or_else(|err| panic!("step graph decode step {p}: {err:#}"));
    assert_eq!(
        graph.current_len(),
        p + 1,
        "the captured step graph must have written exactly one KV row per teacher-forced position, else the row read back is not the row at {p}"
    );
    graph
        .logits_host()
        .unwrap_or_else(|err| panic!("step graph logits at step {p}: {err:#}"))
}

fn new_step_graph_over_a_fresh_ring_cache(
    model: &Arc<Laguna>,
    slots: usize,
) -> LagunaStepGraph {
    let cache = model.new_kv_cache(slots).expect("ring kv cache");
    LagunaStepGraph::new(Arc::clone(model), cache).expect(
        "LagunaStepGraph over the ring cache -- the identical construction laguna_serve.rs performs unconditionally for every served laguna request",
    )
}

fn argmax_index_of_a_full_vocab_row_first_max_wins_on_both_paths(row: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

fn teacher_force_the_prompt_one_token_at_a_time_then_take_a_greedy_chain(
    step_logits_row: &mut dyn FnMut(usize, u32) -> Vec<f32>,
    prompt: &[u32],
    steps: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut row: Vec<f32> = Vec::new();
    for (p, &t) in prompt.iter().enumerate() {
        row = step_logits_row(p, t);
    }
    let prompt_end_row = row.clone();
    let mut chain: Vec<u32> = Vec::with_capacity(steps);
    for i in 0..steps {
        let tok = argmax_index_of_a_full_vocab_row_first_max_wins_on_both_paths(&row);
        chain.push(tok);
        row = step_logits_row(prompt.len() + i, tok);
    }
    (chain, prompt_end_row)
}

#[test]
#[ignore = "loads the Laguna-XS-2.1 NVFP4 checkpoint; set NV_PPL_TEST=1 and NV_PPL_CORPUS; chat-wrapped with the checkpoint's own template (default Poolside system, enable_thinking=false renders <assistant></think>), scored on the cuda decode-stepping path -- NV_LAGUNA_PPL_GRAPH=1 scores through the captured LagunaStepGraph instead, which is the only mode that executes an arm living inside the step graph"]
fn laguna_chat_wrapped_continuation_teacher_forced_ppl_via_serving_decode() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    assert_no_graph_only_arm_is_set_without_the_prerequisite_that_lets_it_engage_because_an_inert_arm_scores_digit_identical_and_reads_as_held();
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let basis = load_the_checkpoint_and_build_the_chat_wrapped_scoring_sequence();
    let LagunaChatPplBasis {
        model,
        device,
        ids,
        score_start,
        label,
    } = basis;

    let graphed = score_through_the_captured_step_graph_that_laguna_serve_builds_unconditionally();
    let path = if graphed {
        "step_graph_capture_LagunaStepGraph"
    } else {
        "decode_stepping_eager_forward_with_cache"
    };
    let family = if graphed {
        "laguna-chat-stepgraph"
    } else {
        "laguna-chat"
    };
    eprintln!(
        "PPL basis: {label} chat_wrapped prompt_tokens={score_start} scored_tokens={} backend=cuda batch=1 path={path} arm={} target_only_no_dflash_drafter",
        ids.len() - score_start,
        arm_label_from_the_resolved_value_of_every_numerics_changing_gate_plus_the_shared_fold_prerequisite()
    );

    let mut shuffled_ids = ids.clone();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled_ids[score_start..],
    );

    let (real, shuffled) = if graphed {
        let slots = ids.len() + 8;
        let mut g = new_step_graph_over_a_fresh_ring_cache(&model, slots);
        let real = score_tail_from_one_full_vocab_f32_row_per_position_skipping_true_tokens_past_the_logit_width(
            &mut |p, t| step_graph_decode_step_logits_row(&mut g, p, t),
            &ids,
            score_start,
        );
        drop(g);
        let mut g = new_step_graph_over_a_fresh_ring_cache(&model, slots);
        let shuffled = score_tail_from_one_full_vocab_f32_row_per_position_skipping_true_tokens_past_the_logit_width(
            &mut |p, t| step_graph_decode_step_logits_row(&mut g, p, t),
            &shuffled_ids,
            score_start,
        );
        drop(g);
        (real, shuffled)
    } else {
        let mut cache = model.new_kv_cache(ids.len() + 8).expect("kv cache");
        let real = score_tail_from_one_full_vocab_f32_row_per_position_skipping_true_tokens_past_the_logit_width(
            &mut |p, t| eager_decode_step_logits_row(&model, &device, &mut cache, p, t),
            &ids,
            score_start,
        );
        let mut cache = model.new_kv_cache(ids.len() + 8).expect("kv cache");
        let shuffled = score_tail_from_one_full_vocab_f32_row_per_position_skipping_true_tokens_past_the_logit_width(
            &mut |p, t| eager_decode_step_logits_row(&model, &device, &mut cache, p, t),
            &shuffled_ids,
            score_start,
        );
        (real, shuffled)
    };

    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        family,
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}

#[test]
#[ignore = "loads the Laguna-XS-2.1 NVFP4 checkpoint; set NV_PPL_TEST=1 and NV_PPL_CORPUS -- teacher-forces this suite's own chat-wrapped prompt one token at a time through BOTH the eager path and the captured LagunaStepGraph, then takes a 128-step greedy chain on each and asserts they are token-identical; that identity is what licenses reading an NV_LAGUNA_PPL_GRAPH=1 score against an eager score, so run it under every arm whose ppl you intend to compare across the two modes"]
fn laguna_eager_and_captured_step_graph_greedy_chains_are_token_identical_which_is_the_licence_for_comparing_their_perplexities(
) {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    assert_no_graph_only_arm_is_set_without_the_prerequisite_that_lets_it_engage_because_an_inert_arm_scores_digit_identical_and_reads_as_held();
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let basis = load_the_checkpoint_and_build_the_chat_wrapped_scoring_sequence();
    let LagunaChatPplBasis {
        model,
        device,
        ids,
        score_start,
        label,
    } = basis;

    let steps = GREEDY_CHAIN_ANCHOR_STEPS_128_IS_THE_LENGTH_OVER_WHICH_EAGER_AND_STEP_GRAPH_MUST_AGREE_TOKEN_FOR_TOKEN_AND_THAT_AGREEMENT_IS_THE_ONLY_LICENCE_FOR_READING_THE_TWO_SCORERS_ON_ONE_SCALE;
    let prompt = &ids[..score_start];
    let slots = prompt.len() + steps + 8;
    let arm = arm_label_from_the_resolved_value_of_every_numerics_changing_gate_plus_the_shared_fold_prerequisite();
    eprintln!(
        "[laguna_ppl_anchor] {label} prompt_tokens={} chain_steps={steps} arm={arm}",
        prompt.len()
    );

    let mut cache = model.new_kv_cache(slots).expect("kv cache");
    let (eager_chain, eager_prompt_end_row) =
        teacher_force_the_prompt_one_token_at_a_time_then_take_a_greedy_chain(
            &mut |p, t| eager_decode_step_logits_row(&model, &device, &mut cache, p, t),
            prompt,
            steps,
        );
    drop(cache);

    let mut g = new_step_graph_over_a_fresh_ring_cache(&model, slots);
    let (graph_chain, graph_prompt_end_row) =
        teacher_force_the_prompt_one_token_at_a_time_then_take_a_greedy_chain(
            &mut |p, t| step_graph_decode_step_logits_row(&mut g, p, t),
            prompt,
            steps,
        );
    drop(g);

    assert_eq!(
        eager_prompt_end_row.len(),
        graph_prompt_end_row.len(),
        "the two paths returned different logit widths"
    );
    let max_abs_row_delta = eager_prompt_end_row
        .iter()
        .zip(graph_prompt_end_row.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[laguna_ppl_anchor] prompt_end_row max_abs_logit_delta={max_abs_row_delta:.6} vocab={}",
        eager_prompt_end_row.len()
    );

    assert_eq!(
        eager_chain.len(),
        graph_chain.len(),
        "chain lengths differ before comparison"
    );
    let first_divergence = eager_chain
        .iter()
        .zip(graph_chain.iter())
        .position(|(a, b)| a != b);
    let differing = eager_chain
        .iter()
        .zip(graph_chain.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "[laguna_ppl_anchor] CHAIN eager arm={arm} {}",
        eager_chain
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    eprintln!(
        "[laguna_ppl_anchor] CHAIN graph arm={arm} {}",
        graph_chain
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    assert!(
        first_divergence.is_none(),
        "eager and captured-step-graph greedy chains diverge at step {:?} ({differing}/{steps} tokens differ, max_abs_logit_delta {max_abs_row_delta:.6} at the prompt end): with this identity broken an NV_LAGUNA_PPL_GRAPH=1 perplexity cannot be read against an eager perplexity, and every arm adjudicated across the two modes has to be re-measured inside one mode",
        first_divergence
    );
}
