#![cfg(feature = "wgpu")]

mod ctx_timing_common;
mod hub_snapshot;

use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu;
use std::time::Instant;

const QAT_REPO: &str = "google/gemma-4-E4B-it-qat-w4a16-ct";
const BF16_REPO: &str = "google/gemma-4-E4B-it";
const SESSION_ROUNDS_2_THE_FIRST_IS_THE_COLD_PROCESS_THE_SECOND_IS_A_WARM_SERVER_REQUEST: usize =
    2;
const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const KV_HEADROOM_128_SLOTS_COVERS_THE_TIMED_DECODE_TAIL: usize = 128;

fn first_ctx_depth_from_env_same_grammar_as_the_cuda_suite_comma_list_each_entry_optional_k_suffix_default_8192(
) -> usize {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => {
            let s = v.split(',').next().expect("split yields at least one entry").trim();
            let (num, mult) = match s.strip_suffix(['k', 'K']) {
                Some(n) => (n, 1024usize),
                None => (s, 1usize),
            };
            num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
        }
        Err(_) => 8192,
    }
}

fn basis_read_from_config_json_quantization_config_not_from_probe_order_because_nv_e4b_dir_may_hold_either_checkpoint(
    dir: &std::path::Path,
) -> &'static str {
    let cfg = std::fs::read_to_string(dir.join("config.json")).unwrap_or_default();
    if cfg.contains("\"quantization_config\"") {
        "w4a16-qat"
    } else {
        "dense-bf16"
    }
}

fn boot_warmup_enabled_same_nv_wgpu_warmup_predicate_as_chat_engine_this_test_warms_only_the_entry_points_it_times(
) -> bool {
    std::env::var("NV_WGPU_WARMUP").ok().as_deref() != Some("0")
}

#[test]
#[ignore = "loads the gemma-4 E4B serving checkpoint on wgpu (qat-w4a16-ct preferred, bf16 \
            fallback -- the ckpt= field says which); set NV_E4B_SERVING_TEST=1 -- walls the \
            wgpu serving session start (chunked prefill_prompt, first decode step, steady \
            decode median) at the first NV_CTX_TOKENS entry (comma list, k suffix ok, default \
            8192), twice in one process: round \
            0 is the cold process, round 1 the warm server request; NV_WGPU_WARMUP=0 disables \
            the serving-default boot warmup pass so round 0 shows the raw cold cost"]
fn e4b_wgpu_serving_session_phase_walls_cold_then_warm() {
    if std::env::var("NV_E4B_SERVING_TEST").ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_E4B_SERVING_TEST=1");
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = hub_snapshot::dir_from_env_or_hub(
        "NV_E4B_DIR",
        QAT_REPO,
        &["config.json", "*.safetensors"],
    )
    .or_else(|| {
        hub_snapshot::dir_from_env_or_hub("NV_E4B_DIR", BF16_REPO, &["config.json", "*.safetensors"])
    })
    .expect("neither E4B snapshot is hydrated; cache the qat-w4a16-ct or bf16 checkpoint");
    let repo = if basis_read_from_config_json_quantization_config_not_from_probe_order_because_nv_e4b_dir_may_hold_either_checkpoint(&dir) == "w4a16-qat" {
        QAT_REPO
    } else {
        BF16_REPO
    };
    let rev = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());
    let ckpt = format!("{repo}@{rev}");
    let depth = first_ctx_depth_from_env_same_grammar_as_the_cuda_suite_comma_list_each_entry_optional_k_suffix_default_8192();

    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    let max_seq = depth + KV_HEADROOM_128_SLOTS_COVERS_THE_TIMED_DECODE_TAIL;
    assert!(max_seq <= config.max_position_embeddings);
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let t0 = Instant::now();
    let mut m = Gemma4E4bWgpu::from_loader(config, &loader, max_seq).expect("build from loader");
    drop(loader);
    let model_build_s = t0.elapsed().as_secs_f64();

    let mut boot_warmup_s = 0.0f64;
    if boot_warmup_enabled_same_nv_wgpu_warmup_predicate_as_chat_engine_this_test_warms_only_the_entry_points_it_times() {
        let t0 = Instant::now();
        let tok = 2u32;
        let chunk = m.prefill_chunk_len();
        if chunk > 0 {
            m.prefill_tokens(&vec![tok; chunk]).expect("warmup prefill chunk");
        }
        m.decode_step(tok).expect("warmup decode step");
        if m.verify_max_rows() >= 2 && m.current_pos() + m.prefill_chunk_len() <= m.max_seq() {
            m.verify_chain(&[tok, tok]).expect("warmup verify chain");
        }
        m.reset();
        boot_warmup_s = t0.elapsed().as_secs_f64();
    }
    eprintln!(
        "e4b wgpu checkpoint: {ckpt} model_build_s={model_build_s:.3} \
         boot_warmup_s={boot_warmup_s:.3} warmup_on={}",
        boot_warmup_enabled_same_nv_wgpu_warmup_predicate_as_chat_engine_this_test_warms_only_the_entry_points_it_times()
    );

    let prompt: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
    for round in 0..SESSION_ROUNDS_2_THE_FIRST_IS_THE_COLD_PROCESS_THE_SECOND_IS_A_WARM_SERVER_REQUEST
    {
        m.reset();
        let t_all = Instant::now();
        let t0 = Instant::now();
        let mut token = m.prefill_prompt(&prompt).expect("chunked prefill_prompt");
        let trunk_prefill_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        token = m.decode_step(token).expect("first decode step");
        let first_step_s = t0.elapsed().as_secs_f64();
        let total_s = t_all.elapsed().as_secs_f64();

        let mut step_ms: Vec<f64> = Vec::new();
        for _ in 0..TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN {
            let t0 = Instant::now();
            token = m.decode_step(token).expect("timed decode step");
            step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms_tok = step_ms[step_ms.len() / 2];

        eprintln!(
            "PREFILL-PHASES-E4B-WGPU depth={depth} round={round} ckpt={ckpt} \
             trunk_prefill_s={trunk_prefill_s:.3} first_step_s={first_step_s:.3} \
             total_s={total_s:.3} trunk_tok_s={:.1} decode_median_ms_tok={median_ms_tok:.3}",
            depth as f64 / trunk_prefill_s,
        );
        if round == 0 {
            println!(
                "CTX-SCALING e4b-wgpu-serving depth={depth} arm=serving steps={} \
                 warmup_steps=1 median_ms_tok={median_ms_tok:.3} tok_s={:.1} \
                 prefill_s={trunk_prefill_s:.2} model_build_s={model_build_s:.2} \
                 boot_warmup_s={boot_warmup_s:.2} checkpoint={ckpt}",
                TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
                1000.0 / median_ms_tok,
            );
        }
    }
}
