#![cfg(feature = "wgpu")]

mod common;
use common::CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE;
mod ppl_common;

use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
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
    common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_E4B_SNAPSHOT",
        &E4B_HOME_HUB_REPOS_W4A16_PACK_QUANTIZED_FIRST_BECAUSE_THE_SPEED_DATA_IS_FROM_THAT_CHECKPOINT,
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
    ppl_common::score_tail_teacher_forced_wgpu_decode_step_logits_full_vocab_row(
        &mut |p, id| {
            m.decode_step_logits(id)
                .unwrap_or_else(|err| panic!("wgpu decode step {p}: {err:#}"))
                .1
        },
        vocab,
        ids,
        score_start,
    )
}

const NEAR_TIE_TOP2_LOGIT_GAP_0P5_IS_WHERE_A_REASSOCIATION_SIZED_PERTURBATION_CAN_FLIP_THE_ARGMAX:
    f32 = 0.5;

const NEAR_TIE_FRACTION_ABOVE_THIS_MEANS_THE_FAMILYS_OWN_SPLIT_LADDER_MUST_BE_MEASURED_BEFORE_ANY_SINGLE_CELL_IS_TRUSTED:
    f64 = 0.10;

fn score_tail_also_recording_every_scored_positions_top2_logit_gap(
    m: &mut Gemma4E4bWgpu,
    ids: &[u32],
    score_start: usize,
) -> (TeacherForcedNllFp32SoftmaxF64Sum, Vec<f32>) {
    let mut nll = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut gaps = Vec::new();
    for p in 0..ids.len() - 1 {
        let row = m
            .decode_step_logits(ids[p])
            .unwrap_or_else(|err| panic!("wgpu decode step {p}: {err:#}"))
            .1;
        if p + 1 < score_start {
            continue;
        }
        let (mut m1, mut m2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &v in &row {
            if v > m1 {
                m2 = m1;
                m1 = v;
            } else if v > m2 {
                m2 = v;
            }
        }
        gaps.push(m1 - m2);
        nll.add_position_full_vocab_row(&row, ids[p + 1]);
    }
    (nll, gaps)
}

const PPL_SPLIT_LADDER_ENV: &str = "NV_PPL_SPLIT_LADDER";

const SPLIT_LADDER_RUNGS_MIN_2_BECAUSE_A_NOISE_BAND_IS_A_RANGE_NOT_A_POINT: usize = 2;

fn split_ladder_rungs_from_env_failing_loudly() -> Vec<u32> {
    let listed = std::env::var(PPL_SPLIT_LADDER_ENV).unwrap_or_else(|_| {
        panic!(
            "{PPL_SPLIT_LADDER_ENV} must list comma-separated split counts: a split count is the \
             same arithmetic in a different reduction order, so the spread of this ladder IS this \
             scorer's own noise floor for this family"
        )
    });
    let rungs: Vec<u32> = listed
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{PPL_SPLIT_LADDER_ENV}={listed}: expected integers"))
        })
        .collect();
    assert!(
        rungs.len() >= SPLIT_LADDER_RUNGS_MIN_2_BECAUSE_A_NOISE_BAND_IS_A_RANGE_NOT_A_POINT,
        "{PPL_SPLIT_LADDER_ENV}={listed} gave {} rungs",
        rungs.len()
    );
    rungs
}

fn ladder_band_line(family: &str, label: &str, ppls: &[f64], accs: &[f64]) {
    let n = ppls.len() as f64;
    let mean = ppls.iter().sum::<f64>() / n;
    let sd = (ppls.iter().map(|p| (p - mean) * (p - mean)).sum::<f64>() / (n - 1.0)).sqrt();
    let lo = ppls.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = ppls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let alo = accs.iter().cloned().fold(f64::INFINITY, f64::min);
    let ahi = accs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "PPL-REASSOCIATION-BAND {family} {label} rungs={} ppl_mean={mean:.3} ppl_sd={sd:.3} \
         ppl_min={lo:.3} ppl_max={hi:.3} ppl_range={:.3} acc_range={:.4}",
        ppls.len(),
        hi - lo,
        ahi - alo
    );
}

const DEEP_BAND_KNEE_64_ENGAGES_OVER_THE_WHOLE_SCORED_RANGE_SO_THE_LADDER_IS_A_NULL_AND_NOT_A_GATE_TEST:
    u32 = 64;

const A_REASSOCIATION_NULL_MAY_NOT_MOVE_TOP1_ACCURACY_BY_MORE_THAN_THIS_OR_THE_SPLIT_COUNT_IS_NOT_THE_SAME_ARITHMETIC:
    f64 = 0.05;

fn split_count_reassociation_ladder(family: &str, rungs: &[u32], arm: &dyn Fn(u32)) {
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
    let label = checkpoint_label_from_snapshot_dir(&dir);

    let (mut ppls, mut accs) = (Vec::new(), Vec::new());
    for &rung in rungs {
        arm(rung);
        let config =
            Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config.json");
        let loader =
            nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
        let mut m = Gemma4E4bWgpu::from_loader(config, &loader, ids.len() + 8).expect("wgpu model");
        drop(loader);
        let (nll, mut gaps) = score_tail_also_recording_every_scored_positions_top2_logit_gap(
            &mut m,
            &ids,
            score_start,
        );
        drop(m);
        assert_eq!(
            gaps.len(),
            nll.scored_positions,
            "the gap census and the canonical nll accumulator must see the same scored positions"
        );
        assert!(
            gaps.iter().all(|g| g.is_finite() && *g >= 0.0),
            "a top-2 logit gap is a difference between the two largest finite logits and cannot \
             be negative or non-finite"
        );
        let near = gaps
            .iter()
            .filter(|g| {
                **g < NEAR_TIE_TOP2_LOGIT_GAP_0P5_IS_WHERE_A_REASSOCIATION_SIZED_PERTURBATION_CAN_FLIP_THE_ARGMAX
            })
            .count();
        let fraction = near as f64 / gaps.len() as f64;
        gaps.sort_by(|a, b| a.partial_cmp(b).expect("finite gaps"));
        let (ppl, acc) = (
            nll.perplexity_exp_of_mean_neg_ln_p(),
            nll.top1_accuracy(),
        );
        eprintln!(
            "PPL-SPLIT-RUNG {family} {label} splits={rung} positions={} ppl={ppl:.3} \
             acc={acc:.4} near_ties_top2_gap_under_0p5={near} near_tie_fraction={fraction:.4} \
             p10_top2_gap={:.3} median_top2_gap={:.3} class={}",
            gaps.len(),
            gaps[gaps.len() / 10],
            gaps[gaps.len() / 2],
            if fraction
                > NEAR_TIE_FRACTION_ABOVE_THIS_MEANS_THE_FAMILYS_OWN_SPLIT_LADDER_MUST_BE_MEASURED_BEFORE_ANY_SINGLE_CELL_IS_TRUSTED
            {
                "LADDER-REQUIRED"
            } else {
                "LADDER-OPTIONAL"
            }
        );
        assert!(
            (acc - accs.first().copied().unwrap_or(acc)).abs()
                <= A_REASSOCIATION_NULL_MAY_NOT_MOVE_TOP1_ACCURACY_BY_MORE_THAN_THIS_OR_THE_SPLIT_COUNT_IS_NOT_THE_SAME_ARITHMETIC,
            "splits={rung} scored top-1 accuracy {acc:.4} against {:.4} on the first rung: a \
             reduction-order change perturbs logits by float noise and can only flip near-ties, so \
             an accuracy move this large means the stage2 is merging a different number of \
             partials than stage1 wrote and this rung is a defect, not a null",
            accs.first().copied().unwrap_or(acc)
        );
        ppls.push(ppl);
        accs.push(acc);
    }
    ladder_band_line(family, &label, &ppls, &accs);
}

#[test]
#[ignore = "builds a real MatFormer E4B checkpoint once per deep-band split count in one process; set NV_PPL_TEST=1, NV_PPL_CORPUS and NV_PPL_SPLIT_LADDER=24,32,47,... -- the deep stage2 is generated for its own split count, so every rung is the same arithmetic in a different reduction order and the spread of this ladder is what one e4b ppl cell cannot resolve"]
fn e4b_split_count_reassociation_ladder_and_top2_gap_density_bound_what_one_ppl_cell_resolves() {
    let rungs = split_ladder_rungs_from_env_failing_loudly();
    let knee =
        DEEP_BAND_KNEE_64_ENGAGES_OVER_THE_WHOLE_SCORED_RANGE_SO_THE_LADDER_IS_A_NULL_AND_NOT_A_GATE_TEST;
    std::env::set_var(
        nv_models::gemma4_e4b_wgpu::E4B_SPLIT_DEPTH_ENV,
        knee.to_string(),
    );
    split_count_reassociation_ladder("e4b-chat-wgpu", &rungs, &|rung| {
        std::env::set_var(
            nv_models::gemma4_e4b_wgpu::E4B_DEEP_SPLITS_ENV,
            rung.to_string(),
        );
        assert_eq!(
            nv_models::gemma4_e4b_wgpu::deep_split_arm(),
            Some((knee, rung)),
            "the deep band must resolve to this rung above a knee the scored range clears, or the \
             ladder measures a gate that never fires instead of a reduction order"
        );
    });
    std::env::remove_var(nv_models::gemma4_e4b_wgpu::E4B_DEEP_SPLITS_ENV);
    std::env::remove_var(nv_models::gemma4_e4b_wgpu::E4B_SPLIT_DEPTH_ENV);
}

const SHALLOW_SPLITS_ENV_IS_THE_DECODE_MERGE_WIDTH_ON_EVERY_LAYER_AND_NOT_A_DEPTH_GATE: &str =
    "NV_WGPU_FLASH_SPLITS";

const SPLIT_DEPTH_0_RETIRES_THE_DEEP_ARM_SO_THIS_LADDER_IS_THE_SHALLOW_MERGE_ALONE: &str = "0";

#[test]
#[ignore = "builds a real MatFormer E4B checkpoint once per SHALLOW split count in one process; set NV_PPL_TEST=1, NV_PPL_CORPUS and NV_PPL_SPLIT_LADDER=1,2,4,8,16,32,64,128 -- the shallow stage2 is generated for its own split count, so every rung is the same arithmetic in a different reduction order and this ladder is the family's reassociation floor on the path serving actually decodes"]
fn e4b_shallow_split_count_reassociation_ladder_is_the_floor_on_the_path_serving_decodes() {
    let rungs = split_ladder_rungs_from_env_failing_loudly();
    std::env::set_var(
        nv_models::gemma4_e4b_wgpu::E4B_SPLIT_DEPTH_ENV,
        SPLIT_DEPTH_0_RETIRES_THE_DEEP_ARM_SO_THIS_LADDER_IS_THE_SHALLOW_MERGE_ALONE,
    );
    split_count_reassociation_ladder("e4b-chat-wgpu-shallow", &rungs, &|rung| {
        std::env::set_var(
            SHALLOW_SPLITS_ENV_IS_THE_DECODE_MERGE_WIDTH_ON_EVERY_LAYER_AND_NOT_A_DEPTH_GATE,
            rung.to_string(),
        );
        assert_eq!(
            nv_models::gemma4_e4b_wgpu::decode_splits(),
            rung,
            "the shallow merge width must resolve to this rung, or the ladder builds one arm \
             every time and reports its own repetition as a null"
        );
        assert!(
            nv_models::gemma4_e4b_wgpu::deep_split_arm().is_none(),
            "the deep arm must be retired for the shallow ladder, or a rung that happens to equal \
             the deep band silently measures the deep stage2 instead"
        );
    });
    std::env::remove_var(SHALLOW_SPLITS_ENV_IS_THE_DECODE_MERGE_WIDTH_ON_EVERY_LAYER_AND_NOT_A_DEPTH_GATE);
    std::env::remove_var(nv_models::gemma4_e4b_wgpu::E4B_SPLIT_DEPTH_ENV);
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

    ppl_common::shuffle_tail_score_both_arms_assert_equal_positions_and_print_machine_line(
        "e4b-chat-wgpu",
        &label,
        &ids,
        score_start,
        &mut |arm_ids| {
            if m.current_pos() != 0 {
                m.reset();
            }
            score_tail_via_wgpu_serving_decode_step_logits_full_vocab(&mut m, arm_ids, score_start)
        },
    );
}
