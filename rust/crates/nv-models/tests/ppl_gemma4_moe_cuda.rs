#![cfg(feature = "cuda")]

mod common;
use common::CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE;
mod ppl_common;

use candle_core::{Device, Tensor};
use nv_models::gemma4_moe::{
    nv_g4moe_cuda_nvfp4_opt_in_because_post_training_quant_failed_the_ppl_band, Gemma4Moe, Gemma4MoeConfig,
};
use ppl_common::{
    checkpoint_label_from_snapshot_dir, corpus_text_from_nv_ppl_corpus_env_failing_loudly,
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control,
    print_machine_line_with_acc_and_assert_real_beats_shuffled,
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip,
    TeacherForcedNllFp32SoftmaxF64Sum,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const G4MOE_HOME_HUB_REPO_MATCHING_THE_WGPU_PPL_ROWS: &str = "models--google--gemma-4-26B-A4B-it";

fn g4moe_snapshot_dir_env_override_then_home_hub_because_devenv_pins_hf_hub_cache_to_a_fixture_cache(
) -> PathBuf {
    if let Ok(d) = std::env::var("NV_G4MOE_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(&home)
        .join(".cache/huggingface/hub")
        .join(G4MOE_HOME_HUB_REPO_MATCHING_THE_WGPU_PPL_ROWS)
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

fn load_g4moe_via_the_serving_from_loader_arm_so_env_selects_the_expert_residency(
    dir: &std::path::Path,
    device: &Device,
) -> Gemma4Moe {
    let config = Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    let loader = nv_weights::WeightLoader::open_dir(dir, device).expect("weights");
    let m = Gemma4Moe::from_loader(config, &loader, device).expect("cuda model via from_loader");
    let nvfp4_layers = m.experts_nvfp4_resident_layer_count();
    let opt_in = nv_g4moe_cuda_nvfp4_opt_in_because_post_training_quant_failed_the_ppl_band();
    if opt_in {
        assert_eq!(
            nvfp4_layers,
            m.config().base.num_hidden_layers,
            "NV_G4MOE_CUDA_NVFP4=1 must hold nvfp4-resident experts on every layer for the real \
             checkpoint; a partial count means the eligibility gate rejected trained dims"
        );
    } else {
        assert_eq!(
            nvfp4_layers, 0,
            "the serving default keeps every expert stack bf16: the nvfp4 arm measured outside \
             the ppl band on this non-QAT checkpoint (rows in perf/runs.jsonl)"
        );
    }
    eprintln!(
        "arm: nvfp4_opt_in={opt_in} nvfp4_resident_layers={nvfp4_layers}/{}",
        m.config().base.num_hidden_layers
    );
    m
}

fn score_tail_via_cuda_serving_decode_step_logits_full_vocab(
    m: &Gemma4Moe,
    ids: &[u32],
    score_start: usize,
) -> TeacherForcedNllFp32SoftmaxF64Sum {
    let vocab = m.config().base.vocab_size;
    let device = m.device().clone();
    let mut cache = m.new_kv_cache(ids.len() + 1).expect("kv cache");
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    for p in 0..ids.len() - 1 {
        let tok = Tensor::from_vec(vec![ids[p]], (1usize, 1usize), &device).unwrap();
        let pos = Tensor::from_vec(vec![p as i32], 1usize, &device).unwrap();
        let logits = m
            .forward_with_cache(&tok, &pos, &mut cache)
            .unwrap_or_else(|err| panic!("cuda decode step {p}: {err:#}"));
        if p + 1 < score_start {
            continue;
        }
        let row: Vec<f32> = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            row.len(),
            vocab,
            "forward_with_cache must return the full vocab row, not a top-k slice"
        );
        acc.add_position_full_vocab_row(&row, ids[p + 1]);
    }
    acc
}

#[test]
#[ignore = "loads the real ~26B-A4B MoE checkpoint on cuda; set NV_PPL_TEST=1 and NV_PPL_CORPUS -- run once per arm: unset for the bf16 serving default, NV_G4MOE_CUDA_NVFP4=1 for the opt-in nvfp4 arm"]
fn g4moe_chat_wrapped_continuation_teacher_forced_ppl_via_cuda_serving_decode() {
    require_nv_ppl_test_gate_because_this_suite_must_never_silently_skip();
    let corpus = corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let dir =
        g4moe_snapshot_dir_env_override_then_home_hub_because_devenv_pins_hf_hub_cache_to_a_fixture_cache();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let (ids, score_start) =
        ppl_common::chat_wrapped_continuation_ids_and_copy_start_rendered_through_the_snapshot_jinja(
            &dir,
            &tok,
            &corpus,
            bos,
            CHAT_WRAPPED_PROMPT_CONTEXT_TOKENS_256_THEN_SCORE_512_SO_EVERY_CHAT_PPL_FAMILY_SHARES_ONE_SLICE,
            CHAT_WRAPPED_SCORED_CONTINUATION_TOKENS,
        );

    let device = Device::new_cuda(0).expect("cuda device");
    let m = load_g4moe_via_the_serving_from_loader_arm_so_env_selects_the_expert_residency(
        &dir, &device,
    );
    let label = checkpoint_label_from_snapshot_dir(&dir);
    eprintln!(
        "PPL basis: {label} chat_wrapped prompt_tokens={score_start} scored_tokens={} backend=cuda batch=1 path=forward_with_cache_seq1 nvfp4_resident_layers={}",
        ids.len() - score_start,
        m.experts_nvfp4_resident_layer_count()
    );

    let real = score_tail_via_cuda_serving_decode_step_logits_full_vocab(&m, &ids, score_start);

    let mut shuffled_ids = ids.clone();
    deterministic_xorshift_fisher_yates_shuffle_fixed_seed_so_every_config_scores_the_same_control(
        &mut shuffled_ids[score_start..],
    );
    let shuffled =
        score_tail_via_cuda_serving_decode_step_logits_full_vocab(&m, &shuffled_ids, score_start);

    assert_eq!(
        real.scored_positions, shuffled.scored_positions,
        "real and shuffled arms scored different position counts"
    );
    print_machine_line_with_acc_and_assert_real_beats_shuffled(
        "g4moe-chat-cuda",
        &label,
        real.scored_positions,
        real.perplexity_exp_of_mean_neg_ln_p(),
        shuffled.perplexity_exp_of_mean_neg_ln_p(),
        real.top1_accuracy(),
    );
}

const DECODE_DEPTH_256_MATCHING_THE_WGPU_SYNTHFILL_SHALLOW_RUNG: usize = 256;
const DECODE_STEPS: usize = 64;
const DECODE_WARMUP_STEPS: usize = 8;

#[test]
#[ignore = "loads the real ~26B-A4B MoE checkpoint on cuda; set NV_G4MOE_CUDA_CTX_TEST=1 -- decode ms/tok at depth for the residency A/B (default bf16 vs NV_G4MOE_CUDA_NVFP4=1)"]
fn g4moe_cuda_decode_ms_per_token_at_depth_256_expert_residency_ab() {
    assert_eq!(
        std::env::var("NV_G4MOE_CUDA_CTX_TEST").ok().as_deref(),
        Some("1"),
        "this gated suite must never silently skip: set NV_G4MOE_CUDA_CTX_TEST=1"
    );
    let dir =
        g4moe_snapshot_dir_env_override_then_home_hub_because_devenv_pins_hf_hub_cache_to_a_fixture_cache();
    let device = Device::new_cuda(0).expect("cuda device");
    let m = load_g4moe_via_the_serving_from_loader_arm_so_env_selects_the_expert_residency(
        &dir, &device,
    );
    let vocab = m.config().base.vocab_size as u64;
    let depth = DECODE_DEPTH_256_MATCHING_THE_WGPU_SYNTHFILL_SHALLOW_RUNG;

    let mut cache = m
        .new_kv_cache(depth + DECODE_WARMUP_STEPS + DECODE_STEPS + 1)
        .expect("kv cache");
    let mut state: u64 = 0x9e3779b97f4a7c15;
    let fill: Vec<u32> = (0..depth)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % (vocab / 2)) as u32
        })
        .collect();
    let fill_start = std::time::Instant::now();
    let tok = Tensor::from_vec(fill, (1usize, depth), &device).unwrap();
    let pos: Vec<i32> = (0..depth as i32).collect();
    let pos = Tensor::from_vec(pos, depth, &device).unwrap();
    let logits = m
        .forward_with_cache(&tok, &pos, &mut cache)
        .expect("prefill");
    let mut next = argmax_of_last_row(&logits);
    let fill_s = fill_start.elapsed().as_secs_f64();

    let mut times_ms: Vec<f64> = Vec::with_capacity(DECODE_STEPS);
    for step in 0..DECODE_WARMUP_STEPS + DECODE_STEPS {
        let p = depth + step;
        let t0 = std::time::Instant::now();
        let tok = Tensor::from_vec(vec![next], (1usize, 1usize), &device).unwrap();
        let pos = Tensor::from_vec(vec![p as i32], 1usize, &device).unwrap();
        let logits = m
            .forward_with_cache(&tok, &pos, &mut cache)
            .unwrap_or_else(|err| panic!("decode step {step}: {err:#}"));
        next = argmax_of_last_row(&logits);
        if step >= DECODE_WARMUP_STEPS {
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
    }
    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times_ms[times_ms.len() / 2];
    println!(
        "CTX-SCALING g4moe-cuda-eager depth={depth} median_ms_tok={median:.3} tok_s={:.1} fill_s={fill_s:.1} nvfp4_layers={} steps={DECODE_STEPS} warmup_steps={DECODE_WARMUP_STEPS}",
        1000.0 / median,
        m.experts_nvfp4_resident_layer_count()
    );
}

fn argmax_of_last_row(logits: &Tensor) -> u32 {
    let seq = logits.dims()[1];
    let row: Vec<f32> = logits
        .narrow(1, seq - 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut best = 0usize;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best as u32
}
