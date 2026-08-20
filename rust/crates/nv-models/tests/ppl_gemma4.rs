#![cfg(feature = "cuda")]

mod ppl_common;

use candle_core::Device;
use nv_models::gemma4::{Gemma4, Gemma4Config};
use nv_weights::{QuantizationConfig, WeightLoader};
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control,
    first_n_corpus_tokens_after_tokenization,
    ppl_block_len_with_debug_override_which_makes_the_run_noncanonical,
    print_machine_line_and_assert_real_beats_shuffled,
    print_machine_line_with_acc_and_assert_real_beats_shuffled,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
    TeacherForcedNllFp32SoftmaxF64Sum,
    PPL_BLOCK_512_TOKENS_FRESH_CACHE_SINGLE_PREFILL_NO_CROSS_BLOCK_CONTEXT_BECAUSE_QWEN36_PREFILL_NLL_DEGRADES_1_9_TO_10_PAST_512_POSITIONS,
    PPL_CORPUS_SLICE_TOKENS_N_2048_CHOSEN_SO_31B_LOAD_PLUS_FOUR_EAGER_512_BLOCKS_FITS_2_TO_3_MIN,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;
mod common;
use common::gemma4_nvfp4_snapshot_dir_env_override_then_home_hub;

fn pin_w4a4_prefill_off_so_ppl_measures_the_model_not_the_prefill_approximation() {
    if std::env::var_os("NV_PREFILL_W4A4").is_none() {
        std::env::set_var("NV_PREFILL_W4A4", "0");
    }
}

fn load_gemma4(dir: &PathBuf, device: &Device) -> Gemma4 {
    pin_w4a4_prefill_off_so_ppl_measures_the_model_not_the_prefill_approximation();
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(dir, device).expect("weights");
    Gemma4::from_loader_quantized(cfg, &weights, &qcfg, device).expect("model")
}

fn score_corpus_in_bos_prefixed_blocks_via_the_reference_full_masked_forward(
    model: &Gemma4,
    device: &Device,
    corpus_tokens: &[u32],
    bos: u32,
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    let block = ppl_block_len_with_debug_override_which_makes_the_run_noncanonical();
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut bstart = 0usize;
    while bstart < corpus_tokens.len() {
        let bend = (bstart + block).min(corpus_tokens.len());
        let mut ids = Vec::with_capacity(bend - bstart + 1);
        ids.push(bos);
        ids.extend_from_slice(&corpus_tokens[bstart..bend]);
        let b = score_block_reference_full_sequence_masked_forward(model, device, &ids);
        eprintln!(
            "PPL-WINDOW start={bstart} scored={} mean_nll={:.3}",
            b.scored_positions,
            b.sum_neg_ln_p_f64 / b.scored_positions as f64
        );
        acc.sum_neg_ln_p_f64 += b.sum_neg_ln_p_f64;
        acc.scored_positions += b.scored_positions;
        bstart = bend;
    }
    acc
}

fn score_block_reference_full_sequence_masked_forward(
    model: &Gemma4,
    device: &Device,
    ids_with_bos: &[u32],
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    let len = ids_with_bos.len();
    let tokens =
        candle_core::Tensor::from_vec(ids_with_bos.to_vec(), (1usize, len), device).expect("tokens");
    let positions = candle_core::Tensor::from_vec((0..len as i32).collect::<Vec<_>>(), len, device)
        .expect("positions");
    let mut fmask = vec![0f32; len * len];
    for i in 0..len {
        for j in 0..=i {
            fmask[i * len + j] = 1.0;
        }
    }
    let fmask_t = candle_core::Tensor::from_vec(fmask, (len, len), device).expect("mask");
    let (logits, _aux) = model
        .forward_with_aux_hidden_masked(&tokens, &positions, &[], Some(&fmask_t))
        .expect("reference masked forward");
    let flat: Vec<f32> = logits
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("host");
    assert_eq!(flat.len() % len, 0, "logit rows not divisible by seq len");
    let vocab = flat.len() / len;
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    for g in 0..len - 1 {
        acc.add_position_full_vocab_row(&flat[g * vocab..(g + 1) * vocab], ids_with_bos[g + 1]);
    }
    acc
}

const CROSS_PATH_DRIFT_BOUND_1_NAT_CATCHES_MISALIGNMENT_NOT_KERNEL_PARITY_MEASURED_DELTA_0_42: f64 =
    1.0;

#[test]
#[ignore = "loads the ~18 GB Gemma-4-31B NVFP4 checkpoint; set NV_PPL_TEST=1 and NV_PPL_CORPUS"]
fn gemma4_cuda_ppl_harness_first_block_within_one_nat_of_reference_full_masked_forward() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = gemma4_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok
        .token_to_id("<bos>")
        .expect("gemma vocab must contain <bos>");
    let n =
        PPL_BLOCK_512_TOKENS_FRESH_CACHE_SINGLE_PREFILL_NO_CROSS_BLOCK_CONTEXT_BECAUSE_QWEN36_PREFILL_NLL_DEGRADES_1_9_TO_10_PAST_512_POSITIONS;
    let slice = first_n_corpus_tokens_after_tokenization(&tok, &corpus, n);
    let device = Device::new_cuda(0).expect("cuda");
    let model = load_gemma4(&dir, &device);

    if let Ok(pstart) = std::env::var("NV_PPL_DEBUG_PROBE_START") {
        let pstart: usize = pstart.parse().expect("NV_PPL_DEBUG_PROBE_START usize");
        let big = first_n_corpus_tokens_after_tokenization(&tok, &corpus, pstart + 64);
        let mut probe = vec![bos];
        probe.extend_from_slice(&big[pstart..pstart + 64]);
        let plen = probe.len();
        let tokens_t =
            candle_core::Tensor::from_vec(probe.clone(), (1usize, plen), &device).unwrap();
        let positions_t =
            candle_core::Tensor::from_vec((0..plen as i32).collect::<Vec<_>>(), plen, &device)
                .unwrap();
        let mut fmask = vec![0f32; plen * plen];
        for i in 0..plen {
            for j in 0..=i {
                fmask[i * plen + j] = 1.0;
            }
        }
        let fmask_t = candle_core::Tensor::from_vec(fmask, (plen, plen), &device).unwrap();
        let (logits, _aux) = model
            .forward_with_aux_hidden_masked(&tokens_t, &positions_t, &[], Some(&fmask_t))
            .expect("probe forward");
        let flat: Vec<f32> = logits
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let vocab = flat.len() / plen;
        for g in 0..plen - 1 {
            let row = &flat[g * vocab..(g + 1) * vocab];
            let t = probe[g + 1] as usize;
            let (mut m, mut am) = (f32::NEG_INFINITY, 0usize);
            for (i, &v) in row.iter().enumerate() {
                if v > m {
                    m = v;
                    am = i;
                }
            }
            let denom: f32 = row.iter().map(|&v| (v - m).exp()).sum();
            let p_true = ((row[t] - m).exp() / denom) as f64;
            let p_top = (1.0 / denom) as f64;
            eprintln!(
                "PPL-PROBE pos={} id={} true={:?} p_true={p_true:.5} top_id={am} top={:?} p_top={p_top:.4}",
                pstart + g,
                probe[g + 1],
                tok.decode(&[probe[g + 1]], false).unwrap_or_default(),
                tok.decode(&[am as u32], false).unwrap_or_default()
            );
        }
        return;
    }

    let via_harness = score_corpus_in_bos_prefixed_blocks_via_the_reference_full_masked_forward(
        &model, &device, &slice, bos,
    );
    let mut ids = Vec::with_capacity(n + 1);
    ids.push(bos);
    ids.extend_from_slice(&slice);
    let via_reference = score_block_reference_full_sequence_masked_forward(&model, &device, &ids);
    let mh = via_harness.sum_neg_ln_p_f64 / via_harness.scored_positions as f64;
    let mr = via_reference.sum_neg_ln_p_f64 / via_reference.scored_positions as f64;
    eprintln!(
        "PPL-PARITY harness_mean_nll={mh:.4} reference_mean_nll={mr:.4} delta={:.4}",
        mh - mr
    );
    assert_eq!(via_harness.scored_positions, via_reference.scored_positions);
    assert!(
        (mh - mr).abs() < CROSS_PATH_DRIFT_BOUND_1_NAT_CATCHES_MISALIGNMENT_NOT_KERNEL_PARITY_MEASURED_DELTA_0_42,
        "harness verify path drifts {:.4} nats from reference full masked forward; \
         a target off-by-one shifts mean nll by ~7 nats, so this is a broken harness",
        mh - mr
    );
}

#[test]
#[ignore = "loads the ~18 GB Gemma-4-31B NVFP4 checkpoint; set NV_PPL_TEST=1 and NV_PPL_CORPUS"]
fn gemma4_cuda_teacher_forced_ppl_bos_prefixed_512_blocks_fresh_cache_no_cross_block_context() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = gemma4_nvfp4_snapshot_dir_env_override_then_home_hub();
    assert!(
        dir.join("model.safetensors.index.json").is_file()
            || dir.join("model.safetensors").is_file(),
        "checkpoint {dir:?} has no weights"
    );
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok
        .token_to_id("<bos>")
        .expect("gemma vocab must contain <bos>");
    let n = PPL_CORPUS_SLICE_TOKENS_N_2048_CHOSEN_SO_31B_LOAD_PLUS_FOUR_EAGER_512_BLOCKS_FITS_2_TO_3_MIN;
    let slice = first_n_corpus_tokens_after_tokenization(&tok, &corpus, n);
    let slice = if std::env::var("NV_PPL_DEBUG_TAIL_ONLY").is_ok() {
        slice[1024..].to_vec()
    } else {
        slice
    };
    let slice = match std::env::var("NV_PPL_DEBUG_N")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(dn) => slice[..dn.min(slice.len())].to_vec(),
        None => slice,
    };

    let device = Device::new_cuda(0).expect("cuda");
    let model = load_gemma4(&dir, &device);
    let label = checkpoint_label_from_snapshot_dir(&dir);
    eprintln!(
        "PPL basis: {label} corpus_slice_tokens={} block=512 backend=cuda batch=1",
        slice.len()
    );

    let real = score_corpus_in_bos_prefixed_blocks_via_the_reference_full_masked_forward(
        &model, &device, &slice, bos,
    );

    let mut shuffled_slice = slice.clone();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled_slice,
    );
    let shuffled = score_corpus_in_bos_prefixed_blocks_via_the_reference_full_masked_forward(
        &model,
        &device,
        &shuffled_slice,
        bos,
    );

    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    print_machine_line_and_assert_real_beats_shuffled(
        "gemma4",
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
    );
}

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

fn score_tail_via_serving_decode_stepping(
    model: &Gemma4,
    device: &Device,
    ids: &[u32],
    score_start: usize,
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    let len = ids.len();
    let mut cache = model.new_kv_cache(len + 8).expect("kv cache");
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    for p in 0..len - 1 {
        let tokens =
            candle_core::Tensor::from_vec(vec![ids[p]], (1usize, 1usize), device).expect("token");
        let positions =
            candle_core::Tensor::from_vec(vec![p as i32], 1usize, device).expect("position");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .unwrap_or_else(|err| panic!("decode step {p}: {err:#}"));
        if p + 1 < score_start {
            continue;
        }
        let row: Vec<f32> = logits
            .to_dtype(candle_core::DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("host");
        acc.add_position_full_vocab_row(&row, ids[p + 1]);
    }
    acc
}

#[test]
#[ignore = "loads the 31B; set NV_PPL_TEST=1 -- the ONLY valid gemma4 PPL: chat-wrapped, scored on the serving decode path (raw-corpus gemma4 PPL is an invalid eval, see #94)"]
fn gemma4_chat_wrapped_continuation_teacher_forced_ppl_via_serving_decode() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = gemma4_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let (ids, score_start) =
        chat_wrapped_continuation_ids_and_copy_start(&dir, &tok, &corpus, bos);
    let device = Device::new_cuda(0).expect("cuda");
    let model = load_gemma4(&dir, &device);
    let label = checkpoint_label_from_snapshot_dir(&dir);
    eprintln!(
        "PPL basis: {label} chat_wrapped prompt_tokens={score_start} scored_tokens={} backend=cuda batch=1 path=decode_stepping",
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
        "gemma4-chat",
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}

fn logits_row_from_masked_forward(
    model: &Gemma4,
    device: &Device,
    ids: &[u32],
    row: usize,
) -> Vec<f32> {
    let len = ids.len();
    let tokens =
        candle_core::Tensor::from_vec(ids.to_vec(), (1usize, len), device).expect("tokens");
    let positions = candle_core::Tensor::from_vec((0..len as i32).collect::<Vec<_>>(), len, device)
        .expect("positions");
    let mut fmask = vec![0f32; len * len];
    for i in 0..len {
        for j in 0..=i {
            fmask[i * len + j] = 1.0;
        }
    }
    let fmask_t = candle_core::Tensor::from_vec(fmask, (len, len), device).expect("mask");
    let (logits, _aux) = model
        .forward_with_aux_hidden_masked(&tokens, &positions, &[], Some(&fmask_t))
        .expect("masked forward");
    let flat: Vec<f32> = logits
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("host");
    let vocab = flat.len() / len;
    flat[row * vocab..(row + 1) * vocab].to_vec()
}

#[test]
#[ignore = "loads the 31B; set NV_PPL_TEST=1 -- the #94 row-position discriminator"]
fn row_p_of_a_long_forward_must_equal_the_last_row_of_the_length_p_forward() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = gemma4_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let toks = first_n_corpus_tokens_after_tokenization(&tok, &corpus, 64);
    let mut ids = vec![bos];
    ids.extend_from_slice(&toks);
    let device = Device::new_cuda(0).expect("cuda");
    let model = load_gemma4(&dir, &device);

    for p in [2usize, 5, 12, 24, 40, 60] {
        let full = logits_row_from_masked_forward(&model, &device, &ids, p);
        let short = logits_row_from_masked_forward(&model, &device, &ids[..p + 1], p);
        let mut worst = 0f32;
        let mut worst_i = 0usize;
        for (i, (a, b)) in full.iter().zip(short.iter()).enumerate() {
            let d = (a - b).abs();
            if d > worst {
                worst = d;
                worst_i = i;
            }
        }
        let am_full = full
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (j, &v)| if v > bv { (j, v) } else { (bi, bv) });
        let am_short = short
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (j, &v)| if v > bv { (j, v) } else { (bi, bv) });
        eprintln!(
            "ROWPROBE p={p} len_full={} worst_abs={worst:.4}@{worst_i} argmax_full={}({:.2}) argmax_short={}({:.2}) argmax_agree={}",
            ids.len(),
            am_full.0,
            am_full.1,
            am_short.0,
            am_short.1,
            am_full.0 == am_short.0
        );
    }
}

#[test]
#[ignore = "loads the 31B; set NV_PPL_TEST=1 -- the #94 layer-localization probe"]
fn the_first_layer_whose_hidden_state_depends_on_future_rows_names_the_defect() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = gemma4_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let toks = first_n_corpus_tokens_after_tokenization(&tok, &corpus, 64);
    let mut ids = vec![bos];
    ids.extend_from_slice(&toks);
    let device = Device::new_cuda(0).expect("cuda");
    let model = load_gemma4(&dir, &device);
    let p = 12usize;
    let aux: Vec<usize> = (0..model.config().num_hidden_layers).collect();

    let run = |ids_slice: &[u32]| -> Vec<Vec<f32>> {
        let len = ids_slice.len();
        let tokens =
            candle_core::Tensor::from_vec(ids_slice.to_vec(), (1usize, len), &device).unwrap();
        let positions =
            candle_core::Tensor::from_vec((0..len as i32).collect::<Vec<_>>(), len, &device)
                .unwrap();
        let mut fmask = vec![0f32; len * len];
        for i in 0..len {
            for j in 0..=i {
                fmask[i * len + j] = 1.0;
            }
        }
        let fmask_t = candle_core::Tensor::from_vec(fmask, (len, len), &device).unwrap();
        let (_logits, aux_h) = model
            .forward_with_aux_hidden_masked(&tokens, &positions, &aux, Some(&fmask_t))
            .expect("forward");
        aux_h
            .iter()
            .map(|h| {
                h.narrow(1, p, 1)
                    .unwrap()
                    .to_dtype(candle_core::DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap()
            })
            .collect()
    };

    let full = run(&ids);
    let short = run(&ids[..p + 1]);
    for (li, (f, sh)) in full.iter().zip(short.iter()).enumerate() {
        let mut worst = 0f32;
        let mut mean = 0f64;
        for (a, b) in f.iter().zip(sh.iter()) {
            let d = (a - b).abs();
            if d > worst {
                worst = d;
            }
            mean += d as f64;
        }
        eprintln!(
            "LAYERPROBE layer={li} worst_abs={worst:.5} mean_abs={:.6}",
            mean / f.len() as f64
        );
    }
}

#[test]
#[ignore = "loads the 31B; set NV_PPL_TEST=1 -- the #94 causality probe"]
fn changing_only_the_last_token_must_not_move_any_earlier_row_else_the_masked_path_leaks_future() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = gemma4_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let toks = first_n_corpus_tokens_after_tokenization(&tok, &corpus, 64);
    let mut ids = vec![bos];
    ids.extend_from_slice(&toks);
    let device = Device::new_cuda(0).expect("cuda");
    let model = load_gemma4(&dir, &device);
    let len = ids.len();
    let aux: Vec<usize> = (0..model.config().num_hidden_layers).collect();

    let run = |ids_slice: &[u32]| -> (Vec<f32>, Vec<Vec<f32>>) {
        let tokens =
            candle_core::Tensor::from_vec(ids_slice.to_vec(), (1usize, len), &device).unwrap();
        let positions =
            candle_core::Tensor::from_vec((0..len as i32).collect::<Vec<_>>(), len, &device)
                .unwrap();
        let mut fmask = vec![0f32; len * len];
        for i in 0..len {
            for j in 0..=i {
                fmask[i * len + j] = 1.0;
            }
        }
        let fmask_t = candle_core::Tensor::from_vec(fmask, (len, len), &device).unwrap();
        let (logits, aux_h) = model
            .forward_with_aux_hidden_masked(&tokens, &positions, &aux, Some(&fmask_t))
            .expect("forward");
        let logits_flat: Vec<f32> = logits
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let hidden: Vec<Vec<f32>> = aux_h
            .iter()
            .map(|h| {
                h.to_dtype(candle_core::DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap()
            })
            .collect();
        (logits_flat, hidden)
    };

    let base = run(&ids);
    let mut perturbed_ids = ids.clone();
    perturbed_ids[len - 1] = if ids[len - 1] == 42 { 43 } else { 42 };
    let pert = run(&perturbed_ids);

    let hidden_width = model.config().hidden_size;
    for (li, (a, b)) in base.1.iter().zip(pert.1.iter()).enumerate() {
        let mut worst = 0f32;
        let mut worst_row = 0usize;
        let mut leaking_rows = 0usize;
        for row in 0..len - 1 {
            let mut row_worst = 0f32;
            for c in 0..hidden_width {
                let d = (a[row * hidden_width + c] - b[row * hidden_width + c]).abs();
                if d > row_worst {
                    row_worst = d;
                }
            }
            if row_worst > 0.0 {
                leaking_rows += 1;
            }
            if row_worst > worst {
                worst = row_worst;
                worst_row = row;
            }
        }
        eprintln!(
            "LEAKPROBE layer={li} leaking_rows={leaking_rows}/{} worst_abs={worst:.6}@row{worst_row}",
            len - 1
        );
    }
    let vocab = base.0.len() / len;
    let mut logit_leaking = 0usize;
    let mut logit_worst = 0f32;
    for row in 0..len - 1 {
        let mut row_worst = 0f32;
        for c in 0..vocab {
            let d = (base.0[row * vocab + c] - pert.0[row * vocab + c]).abs();
            if d > row_worst {
                row_worst = d;
            }
        }
        if row_worst > 0.0 {
            logit_leaking += 1;
        }
        if row_worst > logit_worst {
            logit_worst = row_worst;
        }
    }
    eprintln!(
        "LEAKPROBE logits leaking_rows={logit_leaking}/{} worst_abs={logit_worst:.6}",
        len - 1
    );
    assert_eq!(
        logit_leaking,
        0,
        "rows before the perturbed last token moved: the masked path reads future rows"
    );
}

#[test]
#[ignore = "loads the 31B; set NV_PPL_TEST=1 -- the #94 masked-vs-serving-path discriminator"]
fn masked_forward_rows_must_match_the_unmasked_serving_path_rows() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = gemma4_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let toks = first_n_corpus_tokens_after_tokenization(&tok, &corpus, 64);
    let mut ids = vec![bos];
    ids.extend_from_slice(&toks);
    let device = Device::new_cuda(0).expect("cuda");
    let model = load_gemma4(&dir, &device);
    let len = ids.len();

    let tokens = candle_core::Tensor::from_vec(ids.clone(), (1usize, len), &device).unwrap();
    let positions =
        candle_core::Tensor::from_vec((0..len as i32).collect::<Vec<_>>(), len, &device).unwrap();

    let mut fmask = vec![0f32; len * len];
    for i in 0..len {
        for j in 0..=i {
            fmask[i * len + j] = 1.0;
        }
    }
    let fmask_t = candle_core::Tensor::from_vec(fmask, (len, len), &device).unwrap();
    let (masked_logits, _) = model
        .forward_with_aux_hidden_masked(&tokens, &positions, &[], Some(&fmask_t))
        .expect("masked forward");
    let (unmasked_logits, _) = model
        .forward_with_aux_hidden(&tokens, &positions, &[])
        .expect("unmasked forward");

    let m: Vec<f32> = masked_logits
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let u: Vec<f32> = unmasked_logits
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(m.len(), u.len(), "paths returned different logit shapes");
    let vocab = m.len() / len;
    let mut disagreeing_rows = 0usize;
    for row in 0..len {
        let mr = &m[row * vocab..(row + 1) * vocab];
        let ur = &u[row * vocab..(row + 1) * vocab];
        let mut worst = 0f32;
        for c in 0..vocab {
            let d = (mr[c] - ur[c]).abs();
            if d > worst {
                worst = d;
            }
        }
        let am = |r: &[f32]| {
            r.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (j, &v)| {
                    if v > bv {
                        (j, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0
        };
        let agree = am(mr) == am(ur);
        if !agree {
            disagreeing_rows += 1;
        }
        if row < 8 || row >= len - 3 || !agree {
            eprintln!("PATHPROBE row={row} worst_abs={worst:.4} argmax_agree={agree}");
        }
    }
    eprintln!("PATHPROBE summary argmax_disagreeing_rows={disagreeing_rows}/{len}");
}

#[test]
#[ignore = "loads the 31B; set NV_PPL_TEST=1 -- near-zero NLL copying 512 tokens at depth 600+ proves the whole stack"]
fn scoring_a_verbatim_copy_task_inside_the_chat_template_proves_the_stack_at_depth() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = gemma4_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let wiki = first_n_corpus_tokens_after_tokenization(&tok, &corpus, 512);
    let wiki_text = tok.decode(&wiki, false).expect("decode wiki slice");
    let user =
        format!("Please repeat the following text exactly, with no commentary:\n\n{wiki_text}");
    let prompt = ppl_common::OfficialTemplate::load(&dir).render_user(&user);
    let prompt_ids = tok.encode(prompt.as_str(), false).expect("tokenize prompt");
    let mut ids = prompt_ids.get_ids().to_vec();
    assert_eq!(
        ids.first().copied(),
        Some(bos),
        "the official render must begin with <bos>; hand-prepending it would double-count"
    );
    let copy_start = ids.len();
    ids.extend_from_slice(&wiki);
    let device = Device::new_cuda(0).expect("cuda");
    let model = load_gemma4(&dir, &device);
    let len = ids.len();

    let mut cache = model.new_kv_cache(len + 8).expect("kv cache");
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut copy_argmax_hits = 0usize;
    let mut copy_scored = 0usize;
    for p in 0..len - 1 {
        let tokens = candle_core::Tensor::from_vec(vec![ids[p]], (1usize, 1usize), &device)
            .expect("token");
        let positions =
            candle_core::Tensor::from_vec(vec![p as i32], 1usize, &device).expect("position");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .unwrap_or_else(|err| panic!("decode step {p}: {err:#}"));
        if p + 1 < copy_start {
            continue;
        }
        let row: Vec<f32> = logits
            .to_dtype(candle_core::DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("host");
        acc.add_position_full_vocab_row(&row, ids[p + 1]);
        copy_scored += 1;
        let argmax = row
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (j, &v)| {
                if v > bv {
                    (j, v)
                } else {
                    (bi, bv)
                }
            })
            .0;
        if argmax == ids[p + 1] as usize {
            copy_argmax_hits += 1;
        }
    }
    eprintln!(
        "COPYPROBE copy_start={copy_start} total_len={len} scored={copy_scored} \
         argmax_hits={copy_argmax_hits} mean_nll={:.4} ppl={:.3}",
        acc.sum_neg_ln_p_f64 / acc.scored_positions.max(1) as f64,
        acc.perplexity_exp_of_mean_neg_ln_p()
    );
}

#[test]
#[ignore = "tokenizer-only; set NV_PPL_TEST=1 -- shows the corpus text at the catastrophic decode positions"]
fn show_the_corpus_context_at_the_catastrophic_decode_positions() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = gemma4_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let toks = first_n_corpus_tokens_after_tokenization(&tok, &corpus, 511);
    let mut ids = vec![bos];
    ids.extend_from_slice(&toks);
    for (lo, hi) in [(1usize, 20usize), (80, 104), (190, 204)] {
        let text = tok.decode(&ids[lo..hi], false).unwrap_or_default();
        eprintln!("TOKCTX [{lo}..{hi}] {text:?}");
    }
    for p in [98usize, 99, 100, 101, 102, 199, 200, 201] {
        let piece = tok.decode(&[ids[p]], false).unwrap_or_default();
        eprintln!("TOKCTX tok p={p} id={} {piece:?}", ids[p]);
    }
}

#[test]
#[ignore = "loads the 31B; set NV_PPL_TEST=1 -- fluent greedy text through the decode path contradicts a 7.6-nat model"]
fn greedy_generation_through_the_same_decode_path_shows_whether_the_model_is_actually_broken() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = gemma4_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let mut ids = vec![bos];
    if std::env::var("NV_GREEDY_CHAT").is_ok() {
        let chat = ppl_common::OfficialTemplate::load(&dir)
            .render_user("Write a short paragraph about the actor Robert Boulter and his 2004 role.");
        let enc = tok.encode(chat.as_str(), false).expect("tokenize chat prime");
        assert_eq!(
            enc.get_ids().first().copied(),
            Some(bos),
            "the official render must begin with <bos>; the hand-prepended one would double it"
        );
        ids = enc.get_ids().to_vec();
    } else {
        let toks = first_n_corpus_tokens_after_tokenization(&tok, &corpus, 100);
        ids.extend_from_slice(&toks);
    }
    let device = Device::new_cuda(0).expect("cuda");
    let model = load_gemma4(&dir, &device);

    let prime_len = ids.len();

    fn greedy_via<C: nv_models::gemma4::Gemma4Cache>(
        model: &Gemma4,
        device: &Device,
        ids: &[u32],
        prime_len: usize,
        cache: &mut C,
    ) -> Vec<u32> {
        let mut generated: Vec<u32> = Vec::new();
        let mut next_input = 0u32;
        for p in 0..prime_len + 39 {
            let feed = if p < prime_len { ids[p] } else { next_input };
            let tokens = candle_core::Tensor::from_vec(vec![feed], (1usize, 1usize), device)
                .expect("token");
            let positions =
                candle_core::Tensor::from_vec(vec![p as i32], 1usize, device).expect("position");
            let logits = model
                .forward_with_cache(&tokens, &positions, cache)
                .unwrap_or_else(|err| panic!("decode step {p}: {err:#}"));
            let row: Vec<f32> = logits
                .to_dtype(candle_core::DType::F32)
                .expect("f32")
                .flatten_all()
                .expect("flatten")
                .to_vec1()
                .expect("host");
            let argmax = row
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (j, &v)| {
                    if v > bv {
                        (j, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0 as u32;
            if p >= prime_len - 1 {
                generated.push(argmax);
                next_input = argmax;
            }
        }
        generated
    }

    let generated = if std::env::var("NV_GREEDY_FP8_CACHE").is_ok() {
        let mut cache = model.new_kv_cache_fp8(prime_len + 48).expect("fp8 kv cache");
        greedy_via(&model, &device, &ids, prime_len, &mut cache)
    } else {
        let mut cache = model.new_kv_cache(prime_len + 48).expect("kv cache");
        greedy_via(&model, &device, &ids, prime_len, &mut cache)
    };
    let prime_text = tok.decode(&ids[1..], false).unwrap_or_default();
    let gen_text = tok.decode(&generated, false).unwrap_or_default();
    eprintln!("GREEDY prime ...{:?}", &prime_text[prime_text.len().saturating_sub(120)..]);
    eprintln!("GREEDY continuation {gen_text:?}");
}

#[test]
#[ignore = "loads the 31B; set NV_PPL_TEST=1 -- the #94 ground-truth anchor: serving decode m=1 stepping"]
fn teacher_forced_ppl_via_pure_decode_stepping_is_the_serving_validated_ground_truth() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir = gemma4_nvfp4_snapshot_dir_env_override_then_home_hub();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let toks = first_n_corpus_tokens_after_tokenization(&tok, &corpus, 511);
    let mut ids = vec![bos];
    ids.extend_from_slice(&toks);
    let device = Device::new_cuda(0).expect("cuda");
    let model = load_gemma4(&dir, &device);
    let len = ids.len();

    let mut cache = model.new_kv_cache(len + 8).expect("kv cache");
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    for p in 0..len - 1 {
        let tokens = candle_core::Tensor::from_vec(vec![ids[p]], (1usize, 1usize), &device)
            .expect("token");
        let positions =
            candle_core::Tensor::from_vec(vec![p as i32], 1usize, &device).expect("position");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .unwrap_or_else(|err| panic!("decode step {p}: {err:#}"));
        let row: Vec<f32> = logits
            .to_dtype(candle_core::DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("host");
        acc.add_position_full_vocab_row(&row, ids[p + 1]);
        if p % 64 == 0 {
            eprintln!(
                "DECODE-PPL progress p={p} running_mean_nll={:.3}",
                acc.sum_neg_ln_p_f64 / acc.scored_positions.max(1) as f64
            );
        }
        if matches!(p, 16 | 100 | 200) {
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let z: f64 = row.iter().map(|&v| ((v - mx) as f64).exp()).sum();
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            let show = |i: usize| {
                let word = tok.decode(&[i as u32], false).unwrap_or_default();
                let prob = ((row[i] - mx) as f64).exp() / z;
                format!("{word:?}(logit {:.2}, p {:.4})", row[i], prob)
            };
            let true_id = ids[p + 1] as usize;
            let rank = idx.iter().position(|&i| i == true_id).unwrap_or(usize::MAX);
            eprintln!(
                "DECODE-TOP5 p={p} true={} rank={rank} | {} | {} | {} | {} | {}",
                show(true_id),
                show(idx[0]),
                show(idx[1]),
                show(idx[2]),
                show(idx[3]),
                show(idx[4])
            );
        }
    }
    eprintln!(
        "DECODE-PPL tokens={} scored={} ppl={:.3} mean_nll={:.3}",
        len,
        acc.scored_positions,
        acc.perplexity_exp_of_mean_neg_ln_p(),
        acc.sum_neg_ln_p_f64 / acc.scored_positions as f64
    );
}
