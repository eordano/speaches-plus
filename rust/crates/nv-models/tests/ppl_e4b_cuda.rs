#![cfg(feature = "cuda")]

mod common;
use common::CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE;
mod ppl_common;

use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b::Gemma4E4b;
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control,
    print_machine_line_with_acc_and_assert_real_beats_shuffled,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
    TeacherForcedNllFp32SoftmaxF64Sum,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const E4B_HOME_HUB_REPO_QAT_W4A16_ONLY_BECAUSE_THE_MARLIN_AND_W4A8_ARMS_NEED_PACKED_WEIGHTS:
    [&str; 1] = ["models--google--gemma-4-E4B-it-qat-w4a16-ct"];

const SERVING_PREFILL_CHUNK_512_MATCHES_RUN_SAMPLING_GEMMA4_E4B: usize = 512;

const CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS: usize = 512;

const W4A8_PPL_BAND_2_PCT_IS_THE_FLIPS_REQUIRE_LANDED_PPL_ROWS_BAR_FOR_ACTIVATION_QUANT_ARMS: f64 =
    0.02;

fn e4b_snapshot_dir() -> PathBuf {
    common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_E4B_SNAPSHOT",
        &E4B_HOME_HUB_REPO_QAT_W4A16_ONLY_BECAUSE_THE_MARLIN_AND_W4A8_ARMS_NEED_PACKED_WEIGHTS,
    )
}

fn chat_wrapped_ids_and_score_start(
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

fn drop_both_models_only_after_both_arms_scored_because_the_global_marlin_zero_registry_keeps_raw_persist_pointers_and_a_dropped_models_reused_addresses_get_zeroed_each_step(
    m_off: Gemma4E4b,
    m_on: Gemma4E4b,
) {
    drop(m_off);
    drop(m_on);
}

fn load_model(dir: &std::path::Path, device: &Device) -> Gemma4E4b {
    let cfg = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    assert!(cfg.has_per_layer_embeddings(), "not an E4B checkpoint: {dir:?}");
    let weights = nv_weights::WeightLoader::open_dir(dir, device).expect("weights");
    Gemma4E4b::from_loader(cfg, &weights, device).expect("load Gemma4E4b")
}

fn score_tail_via_cuda_serving_chunk512_prefill_full_logit_rows(
    m: &Gemma4E4b,
    ids: &[u32],
    score_start: usize,
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    assert!(
        !nv_models::gemma4_e4b::step_logits_uncapped_env_opted_in(),
        "this instrument scores the capped serving default; NV_E4B_STEP_LOGITS_UNCAPPED=1 \
         restores the pre-fix raw rows whose family ppl read 705-vs-16 against wgpu and its \
         rows must never land as canonical"
    );
    let n_layers = m.config().num_hidden_layers;
    let vocab = m.config().vocab_size;
    let mut cache: Vec<Option<(Tensor, Tensor)>> = vec![None; n_layers];
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut per_pos: Vec<(usize, f64)> = Vec::new();
    let chunk = std::env::var("NV_E4B_PPL_DEBUG_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(SERVING_PREFILL_CHUNK_512_MATCHES_RUN_SAMPLING_GEMMA4_E4B);
    let mut off = 0usize;
    while off < ids.len() {
        let c = (ids.len() - off).min(chunk);
        let (logits, _hidden) = m
            .forward_step_spec(&ids[off..off + c], off, &mut cache)
            .unwrap_or_else(|e| panic!("prefill chunk at {off}: {e:#}"));
        let need_rows = (off..off + c).any(|p| p + 1 >= score_start && p + 1 < ids.len());
        if need_rows {
            let rows: Vec<Vec<f32>> = logits
                .to_dtype(DType::F32)
                .expect("f32 logits")
                .to_vec2()
                .expect("host rows");
            assert_eq!(rows[0].len(), vocab, "forward_step_spec must return full vocab rows");
            for (i, row) in rows.iter().enumerate() {
                let p = off + i;
                if p + 1 >= score_start && p + 1 < ids.len() {
                    let before = acc.sum_neg_ln_p_f64;
                    acc.add_position_full_vocab_row(row, ids[p + 1]);
                    per_pos.push((p, acc.sum_neg_ln_p_f64 - before));
                }
            }
        }
        off += c;
    }
    per_pos.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("finite nll"));
    let worst: Vec<String> = per_pos
        .iter()
        .take(8)
        .map(|(p, nll)| format!("p={p}:{nll:.2}"))
        .collect();
    eprintln!("worst-position nll: {}", worst.join(" "));
    acc
}

fn score_real_and_shuffled(
    m: &Gemma4E4b,
    ids: &[u32],
    score_start: usize,
) -> (TeacherForcedNllFp32SoftmaxF64Sum, TeacherForcedNllFp32SoftmaxF64Sum) {
    let real = score_tail_via_cuda_serving_chunk512_prefill_full_logit_rows(m, ids, score_start);
    let mut shuffled_ids = ids.to_vec();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled_ids[score_start..],
    );
    let shuffled =
        score_tail_via_cuda_serving_chunk512_prefill_full_logit_rows(m, &shuffled_ids, score_start);
    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    (real, shuffled)
}

#[test]
#[ignore = "loads the gemma-4 E4B qat-w4a16-ct checkpoint on cuda TWICE in one process (the \
            same-session control the flips-require-landed-ppl-rows rule demands); set \
            NV_PPL_TEST=1 and NV_PPL_CORPUS -- scores the shared chat-wrapped slice through \
            the serving chunk-512 prefill path (forward_step_spec full logit rows, the m=512 \
            regime where the a8 GEMMs actually run), once with the a16 serving default and \
            once with NV_E4B_MLP_W4A8FP8=1, prints one PPL machine line per arm plus shuffled \
            controls, and asserts the arm ppl within 2% of the same-session a16 control"]
fn e4b_chat_wrapped_chunked_prefill_ppl_mlp_w4a8fp8_arm_vs_same_session_a16_control() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = e4b_snapshot_dir();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let (ids, score_start) = chat_wrapped_ids_and_score_start(&dir, &tok, &corpus, bos);
    let label = checkpoint_label_from_snapshot_dir(&dir);
    let device = Device::new_cuda_with_stream(0).expect("cuda");

    let a8_env = nv_models::gemma4_e4b::NV_E4B_MLP_W4A8FP8_ENV_OPT_IN_1_RUNS_MLP_MARLIN_GEMMS_WITH_PER_ROW_FP8_ACTIVATIONS_AND_X512_COMPENSATED_SCALES_BECAUSE_CHUNK_512_PREFILL_IS_THE_LARGE_M_REGIME_WHERE_FP8_TENSOR_CORES_PAY_WHILE_DECODE_M1_KEEPS_THE_GEMV_PATH_UNTOUCHED_AND_ATTN_PROJECTIONS_STAY_A16_PER_THE_QWEN_W4A8_MLP_ONLY_PRECEDENT;
    std::env::remove_var(a8_env);
    assert!(
        !nv_models::gemma4_e4b::mlp_w4a8fp8_env_opted_in(),
        "control arm must load with the a16 serving default"
    );
    let m_off = load_model(&dir, &device);
    eprintln!(
        "PPL basis: {label} chat_wrapped prompt_tokens={score_start} scored_tokens={} \
         backend=cuda batch=1 path=forward_step_spec_chunk512 arm=a16-control",
        ids.len() - score_start
    );
    let (real_off, shuffled_off) = score_real_and_shuffled(&m_off, &ids, score_start);

    std::env::set_var(a8_env, "1");
    assert!(
        nv_models::gemma4_e4b::mlp_w4a8fp8_env_opted_in(),
        "arm must load with NV_E4B_MLP_W4A8FP8=1"
    );
    let m_on = load_model(&dir, &device);
    eprintln!(
        "PPL basis: {label} chat_wrapped prompt_tokens={score_start} scored_tokens={} \
         backend=cuda batch=1 path=forward_step_spec_chunk512 arm=mlp-w4a8fp8",
        ids.len() - score_start
    );
    let (real_on, shuffled_on) = score_real_and_shuffled(&m_on, &ids, score_start);
    std::env::remove_var(a8_env);
    drop_both_models_only_after_both_arms_scored_because_the_global_marlin_zero_registry_keeps_raw_persist_pointers_and_a_dropped_models_reused_addresses_get_zeroed_each_step(
        m_off, m_on,
    );

    let ppl_off = real_off.perplexity_exp_of_mean_neg_ln_p();
    let ppl_on = real_on.perplexity_exp_of_mean_neg_ln_p();
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        "e4b-chat-cuda-prefill512",
        &label,
        real_off.scored_positions,
        ppl_off,
        shuffled_off.perplexity_exp_of_mean_neg_ln_p(),
        real_off.top1_accuracy(),
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        "e4b-chat-cuda-prefill512-w4a8fp8",
        &label,
        real_on.scored_positions,
        ppl_on,
        shuffled_on.perplexity_exp_of_mean_neg_ln_p(),
        real_on.top1_accuracy(),
    );
    let rel = (ppl_on - ppl_off).abs() / ppl_off;
    let nats = (ppl_on / ppl_off).ln();
    eprintln!(
        "w4a8fp8 ppl band: control={ppl_off:.4} arm={ppl_on:.4} rel={rel:.4} delta_nats={nats:.5}"
    );
    assert!(
        rel <= W4A8_PPL_BAND_2_PCT_IS_THE_FLIPS_REQUIRE_LANDED_PPL_ROWS_BAR_FOR_ACTIVATION_QUANT_ARMS,
        "mlp w4a8-fp8 moved chat-slice ppl by {:.2}% (control {ppl_off:.4} -> arm {ppl_on:.4}), \
         outside the 2% band that activation-quant arms must hold; the arm stays opt-in-red \
         and must not be advertised as quality-neutral",
        rel * 100.0
    );
}

const PFFLASH_PPL_BAND_2_PCT_MATCHES_THE_ACTIVATION_QUANT_BAR_BECAUSE_HALF_PRECISION_ATTENTION_INPUT_ROUNDING_IS_THE_SAME_KIND_OF_LOSSY:
    f64 = 0.02;

#[test]
#[ignore = "loads the gemma-4 E4B qat-w4a16-ct checkpoint on cuda ONCE (the flash-attn env is \
            reread every call, so one model serves both arms as the same-session control the \
            flips-require-landed-ppl-rows rule demands); set NV_PPL_TEST=1 and NV_PPL_CORPUS \
            -- scores the shared chat-wrapped slice through the serving chunk-512 prefill path \
            (forward_step_spec full logit rows), once with the eager f32 SDPA serving default \
            and once with NV_E4B_PREFILL_FLASH_ATTN=1 (sliding layers hd256 via \
            candle-flash-attn f16, full layers hd512 via bf16 tensor-core GEMMs with fused \
            softmax), prints one PPL machine line per arm plus \
            shuffled controls, and asserts the arm ppl within 2% of the same-session eager \
            control"]
fn e4b_chat_wrapped_chunked_prefill_ppl_flash_attn_arm_vs_same_session_eager_control() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = e4b_snapshot_dir();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let (ids, score_start) = chat_wrapped_ids_and_score_start(&dir, &tok, &corpus, bos);
    let label = checkpoint_label_from_snapshot_dir(&dir);
    let device = Device::new_cuda_with_stream(0).expect("cuda");

    let flash_env = nv_models::gemma4_e4b::NV_E4B_PREFILL_FLASH_ATTN_ENV_OPT_IN_1_SLIDING_LAYERS_HD256_TAKE_CANDLE_FLASH_ATTN_F16_NATIVE_GQA_WINDOW_LEFT_511_AND_FULL_LAYERS_HD512_TAKE_BF16_TENSOR_CORE_GEMMS_WITH_FUSED_SOFTMAX_BECAUSE_THE_HDIM512_FLASH_FWD_KERNEL_RETURNS_WRONG_MASS_ON_SM120_FOR_EVERY_DTYPE_AND_WINDOW_MODE_WHILE_DECODE_N1_KEEPS_ATTN_DECODE_F32_UNTOUCHED;
    std::env::remove_var(flash_env);
    assert!(
        !nv_models::gemma4_e4b::prefill_flash_attn_env_opted_in(),
        "control arm must score with the eager serving default"
    );
    let m = load_model(&dir, &device);
    eprintln!(
        "PPL basis: {label} chat_wrapped prompt_tokens={score_start} scored_tokens={} \
         backend=cuda batch=1 path=forward_step_spec_chunk512 arm=eager-control",
        ids.len() - score_start
    );
    let (real_off, shuffled_off) = score_real_and_shuffled(&m, &ids, score_start);

    std::env::set_var(flash_env, "1");
    assert!(
        nv_models::gemma4_e4b::prefill_flash_attn_env_opted_in(),
        "arm must score with NV_E4B_PREFILL_FLASH_ATTN=1"
    );
    eprintln!(
        "PPL basis: {label} chat_wrapped prompt_tokens={score_start} scored_tokens={} \
         backend=cuda batch=1 path=forward_step_spec_chunk512 arm=pfflash",
        ids.len() - score_start
    );
    let (real_on, shuffled_on) = score_real_and_shuffled(&m, &ids, score_start);
    std::env::remove_var(flash_env);

    let ppl_off = real_off.perplexity_exp_of_mean_neg_ln_p();
    let ppl_on = real_on.perplexity_exp_of_mean_neg_ln_p();
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        "e4b-chat-cuda-prefill512",
        &label,
        real_off.scored_positions,
        ppl_off,
        shuffled_off.perplexity_exp_of_mean_neg_ln_p(),
        real_off.top1_accuracy(),
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        "e4b-chat-cuda-prefill512-pfflash",
        &label,
        real_on.scored_positions,
        ppl_on,
        shuffled_on.perplexity_exp_of_mean_neg_ln_p(),
        real_on.top1_accuracy(),
    );
    let rel = (ppl_on - ppl_off).abs() / ppl_off;
    let nats = (ppl_on / ppl_off).ln();
    eprintln!(
        "pfflash ppl band: control={ppl_off:.4} arm={ppl_on:.4} rel={rel:.4} delta_nats={nats:.5}"
    );
    assert!(
        rel <= PFFLASH_PPL_BAND_2_PCT_MATCHES_THE_ACTIVATION_QUANT_BAR_BECAUSE_HALF_PRECISION_ATTENTION_INPUT_ROUNDING_IS_THE_SAME_KIND_OF_LOSSY,
        "flash-attn prefill moved chat-slice ppl by {:.2}% (control {ppl_off:.4} -> arm \
         {ppl_on:.4}), outside the 2% band; the arm stays opt-in-red and must not be \
         advertised as quality-neutral",
        rel * 100.0
    );
}
