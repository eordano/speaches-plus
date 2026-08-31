#![cfg(feature = "wgpu")]

mod common;
use common::CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE;
mod ppl_common;

use nv_models::gpt_oss_wgpu::{GptOssConfig, GptOssWgpu};
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    first_n_corpus_tokens_after_tokenization,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
    TeacherForcedNllFp32SoftmaxF64Sum,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn gptoss_mxfp4_snapshot_dir_env_override_then_home_hub_because_devenv_pins_hf_hub_cache_to_a_fixture_cache(
) -> PathBuf {
    common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_GPTOSS_SNAPSHOT",
        &["models--openai--gpt-oss-20b"],
    )
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

const SLIDING_AB_GATE_ENV: &str = "NV_GOW_SLIDING_AB_TEST";

const SLIDING_AB_REASSOCIATION_NULL_SPLITS_32_IS_THE_SAME_MATH_AS_THE_DEFAULT_64_SO_ITS_PPL_MOVE_IS_PURE_INSTRUMENT_NOISE: &str =
    "32";

const SLIDING_AB_SWAP_MAY_NOT_EXCEED_THIS_MULTIPLE_OF_THE_REASSOCIATION_NULL_OR_IT_IS_A_REAL_QUALITY_COST: f64 =
    3.0;

struct PerPositionScore {
    ln_p: Vec<f64>,
    argmax: Vec<u32>,
    top2_gap: Vec<f32>,
}

impl PerPositionScore {
    fn ppl(&self) -> f64 {
        (-self.ln_p.iter().sum::<f64>() / self.ln_p.len() as f64).exp()
    }

    fn top1_accuracy(&self, ids: &[u32], score_start: usize) -> f64 {
        let hits = self
            .argmax
            .iter()
            .enumerate()
            .filter(|(k, a)| **a == ids[score_start + k])
            .count();
        hits as f64 / self.argmax.len() as f64
    }
}

fn score_arm_recording_every_scored_position(
    m: &mut GptOssWgpu,
    ids: &[u32],
    score_start: usize,
) -> PerPositionScore {
    let mut out = PerPositionScore {
        ln_p: Vec::new(),
        argmax: Vec::new(),
        top2_gap: Vec::new(),
    };
    for p in 0..ids.len() - 1 {
        let row = m
            .decode_step_logits(ids[p])
            .unwrap_or_else(|err| panic!("wgpu decode step {p}: {err:#}"))
            .1;
        if p + 1 < score_start {
            continue;
        }
        let (mut m1, mut m2, mut am) = (f32::NEG_INFINITY, f32::NEG_INFINITY, 0usize);
        for (j, &v) in row.iter().enumerate() {
            if v > m1 {
                m2 = m1;
                m1 = v;
                am = j;
            } else if v > m2 {
                m2 = v;
            }
        }
        let mut denom = 0f32;
        for &v in &row {
            denom += (v - m1).exp();
        }
        out.ln_p
            .push(((row[ids[p + 1] as usize] - m1) - denom.ln()) as f64);
        out.argmax.push(am as u32);
        out.top2_gap.push(m1 - m2);
    }
    out
}

#[test]
#[ignore = "three real-weights builds of the ~13 GB gpt-oss-20b checkpoint in one process; set NV_GOW_SLIDING_AB_TEST=1 and NV_PPL_CORPUS -- prices the flash-sliding kernel swap against this scorer's own reassociation noise"]
fn the_flash_sliding_kernel_swap_moves_ppl_less_than_a_pure_reassociation_of_the_same_default_does() {
    if std::env::var(SLIDING_AB_GATE_ENV).is_err() {
        eprintln!("skipped: set {SLIDING_AB_GATE_ENV}=1 -- this suite builds the real checkpoint three times");
        return;
    }
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
    let label = checkpoint_label_from_snapshot_dir(&dir);

    let mut arm = |sliding: &str, splits: Option<&str>| {
        std::env::set_var(nv_models::gpt_oss_wgpu::FLASH_SLIDING_ENV, sliding);
        match splits {
            Some(s) => std::env::set_var("NV_GPTOSS_WGPU_FLASH_SPLITS", s),
            None => std::env::remove_var("NV_GPTOSS_WGPU_FLASH_SPLITS"),
        }
        let config = GptOssConfig::from_hf_json_file(dir.join("config.json")).expect("config.json");
        let loader =
            nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
        let mut m = GptOssWgpu::from_loader(config, &loader, ids.len() + 8).expect("wgpu model");
        drop(loader);
        let passes = m.pass_count();
        let s = score_arm_recording_every_scored_position(&mut m, &ids, score_start);
        drop(m);
        (s, passes)
    };

    let (flash, flash_passes) = arm("1", None);
    let (serial, serial_passes) = arm("0", None);
    let (reassoc, reassoc_passes) = arm(
        "1",
        Some(SLIDING_AB_REASSOCIATION_NULL_SPLITS_32_IS_THE_SAME_MATH_AS_THE_DEFAULT_64_SO_ITS_PPL_MOVE_IS_PURE_INSTRUMENT_NOISE),
    );
    std::env::remove_var(nv_models::gpt_oss_wgpu::FLASH_SLIDING_ENV);

    assert!(
        serial_passes < flash_passes && reassoc_passes == flash_passes,
        "the sliding arm must engage: turning the flash sliding kernel off retires one pass per \
         sliding layer ({flash_passes} -> {serial_passes}) and a split-count change retires none \
         ({reassoc_passes}); equal counts mean an arm never took effect"
    );
    let n = flash.ln_p.len();
    assert!(
        n == serial.ln_p.len() && n == reassoc.ln_p.len() && n > 0,
        "every arm must score the same {n} positions"
    );
    let flips = |a: &PerPositionScore, b: &PerPositionScore| {
        (0..n).filter(|&i| a.argmax[i] != b.argmax[i]).count()
    };
    let worst = |a: &PerPositionScore, b: &PerPositionScore| {
        (0..n).fold(0f64, |w, i| w.max((a.ln_p[i] - b.ln_p[i]).abs()))
    };
    let near_ties = flash.top2_gap.iter().filter(|g| **g < 0.5).count();
    let swap = (flash.ppl() - serial.ppl()).abs();
    let null = (flash.ppl() - reassoc.ppl()).abs();
    eprintln!(
        "GOW-SLIDING-AB {label} positions={n} near_ties_top2_gap_under_0p5={near_ties} \
         flash_ppl={:.3} serial_ppl={:.3} reassoc_splits32_ppl={:.3} \
         swap_delta={swap:.3} reassociation_null_delta={null:.3} \
         flips_swap={} flips_null={} worst_dlnp_swap={:.3} worst_dlnp_null={:.3}",
        flash.ppl(),
        serial.ppl(),
        reassoc.ppl(),
        flips(&flash, &serial),
        flips(&flash, &reassoc),
        worst(&flash, &serial),
        worst(&flash, &reassoc),
    );
    assert!(
        swap <= SLIDING_AB_SWAP_MAY_NOT_EXCEED_THIS_MULTIPLE_OF_THE_REASSOCIATION_NULL_OR_IT_IS_A_REAL_QUALITY_COST
            * null,
        "the flash sliding kernel and the serial sliding kernel compute the same masked softmax \
         and differ only in reduction order, so their ppl gap ({swap:.3}) must stay within \
         {SLIDING_AB_SWAP_MAY_NOT_EXCEED_THIS_MULTIPLE_OF_THE_REASSOCIATION_NULL_OR_IT_IS_A_REAL_QUALITY_COST}x \
         the gap this same scorer shows between two reduction orders of the shipped default \
         ({null:.3}); a larger gap is a defect in the kernel, not noise in the instrument"
    );
}

const PAIRED_SLICES_ENV: &str = "NV_GOW_PPL_CORPUS_SLICES";

const PAIRED_SLICES_MIN_3_SO_A_MEAN_CARRIES_A_STANDARD_ERROR_AND_A_SIGN_COUNT: usize = 3;

const PAIRED_NULL_SPLITS_32_IS_THE_SAME_ARITHMETIC_IN_A_DIFFERENT_REDUCTION_ORDER_SO_ITS_PAIRED_MEAN_IS_THIS_SCORERS_OWN_FLOOR: &str =
    "32";

const PAIRED_ABS_T_2_IS_THE_SMALLEST_PAIRED_SLICE_T_THAT_RESOLVES_AN_ARM_FROM_THIS_SCORERS_REASSOCIATION_NOISE: f64 =
    2.0;

const FLASH_SPLITS_ENV: &str = "NV_GPTOSS_WGPU_FLASH_SPLITS";

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

    ppl_common::shuffle_tail_score_both_arms_assert_equal_positions_and_print_machine_line(
        "gptoss-chat-wgpu",
        &label,
        &ids,
        score_start,
        &mut |arm_ids| {
            if m.current_pos() != 0 {
                m.reset().expect("reset between arms");
            }
            score_tail_via_wgpu_serving_decode_step_logits_full_vocab(&mut m, arm_ids, score_start)
        },
    );
}
