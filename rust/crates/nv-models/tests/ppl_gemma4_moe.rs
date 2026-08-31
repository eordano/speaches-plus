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

const NEAR_TIE_TOP2_LOGIT_GAP_0P5_IS_WHERE_A_REASSOCIATION_SIZED_PERTURBATION_CAN_FLIP_THE_ARGMAX:
    f32 = 0.5;

const NEAR_TIE_FRACTION_ABOVE_THIS_MEANS_THE_FAMILYS_OWN_SPLIT_LADDER_MUST_BE_MEASURED_BEFORE_ANY_SINGLE_CELL_IS_TRUSTED:
    f64 = 0.10;

fn score_tail_also_recording_every_scored_positions_top2_logit_gap(
    m: &mut Gemma4MoeWgpu,
    ids: &[u32],
    score_start: usize,
) -> (TeacherForcedNllFp32SoftmaxF64Sum, Vec<f32>) {
    let mut nll = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut gaps = Vec::new();
    for p in 0..ids.len() - 1 {
        let (_argmax_token, row) = m
            .decode_step_logits(ids[p])
            .unwrap_or_else(|err| panic!("wgpu decode step {p}: {err:#}"));
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

const G4MOE_FLASH_SPLITS_ENV: &str = "NV_G4MOE_FLASH_SPLITS";

const A_REASSOCIATION_NULL_MAY_NOT_MOVE_TOP1_ACCURACY_BY_MORE_THAN_THIS_OR_THE_SPLIT_COUNT_IS_NOT_THE_SAME_ARITHMETIC:
    f64 = 0.05;

#[test]
#[ignore = "builds the real ~26B-A4B MoE checkpoint once per split count in one process; set NV_PPL_TEST=1, NV_PPL_CORPUS and NV_PPL_SPLIT_LADDER=4,8,16,... -- a split count is the same arithmetic in a different reduction order, so the spread of this ladder is what one g4moe ppl cell cannot resolve"]
fn g4moe_split_count_reassociation_ladder_and_top2_gap_density_bound_what_one_ppl_cell_resolves() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
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
    let label = checkpoint_label_from_snapshot_dir(&dir);

    let (mut ppls, mut accs) = (Vec::new(), Vec::new());
    for &rung in &rungs {
        std::env::set_var(G4MOE_FLASH_SPLITS_ENV, rung.to_string());
        assert_eq!(
            nv_models::gemma4_moe_wgpu::flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build(),
            rung,
            "the split count must reach the decode flash grid, or the ladder measures nothing"
        );
        let config =
            Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config.json");
        let loader =
            nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
        let mut m = Gemma4MoeWgpu::from_loader(config, &loader, ids.len() + 8).expect("wgpu model");
        drop(loader);
        let flash1_passes = m
            .pass_rows()
            .iter()
            .filter(|(l, _, _, _)| l.starts_with("g4m-at-flash1"))
            .count();
        assert!(
            flash1_passes > 0,
            "the scored decode must dispatch the split flash decode, or a split count changes \
             nothing and this ladder is not a reassociation null"
        );
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
        let (ppl, acc) = (nll.perplexity_exp_of_mean_neg_ln_p(), nll.top1_accuracy());
        eprintln!(
            "PPL-SPLIT-RUNG g4moe-chat-wgpu {label} splits={rung} positions={} ppl={ppl:.3} \
             acc={acc:.4} flash1_passes={flash1_passes} \
             near_ties_top2_gap_under_0p5={near} near_tie_fraction={fraction:.4} \
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
    std::env::remove_var(G4MOE_FLASH_SPLITS_ENV);
    let n = ppls.len() as f64;
    let mean = ppls.iter().sum::<f64>() / n;
    let sd = (ppls.iter().map(|p| (p - mean) * (p - mean)).sum::<f64>() / (n - 1.0)).sqrt();
    let lo = ppls.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = ppls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let alo = accs.iter().cloned().fold(f64::INFINITY, f64::min);
    let ahi = accs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "PPL-REASSOCIATION-BAND g4moe-chat-wgpu {label} rungs={} ppl_mean={mean:.3} ppl_sd={sd:.3} \
         ppl_min={lo:.3} ppl_max={hi:.3} ppl_range={:.3} acc_range={:.4}",
        ppls.len(),
        hi - lo,
        ahi - alo
    );
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
    let weights_decision = nv_models::gemma4_gguf::opted_in_qat_q4_gguf_for_snapshot(&config);
    let mut m = match &weights_decision {
        Some(gguf) => {
            eprintln!(
                "weights arm: OPT-IN qat q4 gguf {} ({}=1)",
                gguf.display(),
                nv_models::gemma4_gguf::G4MOE_SERVE_QAT_Q4_OPT_IN_ENV
            );
            Gemma4MoeWgpu::from_gguf(gguf, ids.len() + 8).expect("wgpu model from qat q4 gguf")
        }
        None => {
            eprintln!("weights arm: snapshot safetensors (the serving default)");
            let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
                .expect("weights");
            Gemma4MoeWgpu::from_loader(config, &loader, ids.len() + 8).expect("wgpu model")
        }
    };

    let mut label = checkpoint_label_from_snapshot_dir(&dir);
    if weights_decision.is_some() {
        label.push_str("+qat-ud-q4-k-xl-gguf");
    }
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
