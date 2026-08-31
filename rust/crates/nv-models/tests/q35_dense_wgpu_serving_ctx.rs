#![cfg(feature = "wgpu")]

mod common;
use common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights;

mod ctx_timing_common;
use ctx_timing_common::median_ms_of_timed_steps;
use ctx_timing_common::nv_ctx_tokens_env_with_k_suffix;
use ctx_timing_common::print_wgpu_profile_share_table_then_reset;

pub const THE_Q35_DENSE_WGPU_SERVING_DECODE_IS_EAGER_DECODE_STEP_DEVICE_ARGMAX_AND_THE_ENGINE_ARMS_NOTHING: &str =
    "rust/src/oapi/chat_engine_wgpu.rs builds this family through \
     Qwen3_5DenseWgpu::from_loader_with_lora with no env armed at build, and its greedy \
     generation loop feeds every token back through Decoder::step -> \
     Qwen3_5DenseWgpu::decode_step: device argmax, no host logits row (the vocab download \
     happens only when the sampler needs logits). The spec chain route never admits this \
     family by default -- spec::SPEC_KINDS_DEFAULT_IS_GEMMA4_E4B_ALONE_BECAUSE_A_VERIFY_FORWARD_\
     IS_NOT_A_MEASURED_WIN, with NV_Q3D_MTP=1 as the separate opt-in -- and cuda serving for \
     the family is itself opt-in (backend_select::QWEN35_DENSE_NO_CUDA), so this eager wgpu \
     decode_step IS the served shape and the bare arm of this ladder times exactly what a \
     default deployment runs.";

const Q35_DENSE_WGPU_SERVES_THE_IG1_NVFP4_SNAPSHOT_BF16_BASE_IS_THE_ESCAPE_ARM: [&str; 2] = [
    "models--ig1--Qwen3.5-9B-NVFP4",
    "models--Qwen--Qwen3.5-9B",
];

const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK: usize =
    ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
        + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
        + 16;

const DEPTH_196608_THE_CONFORMANCE_RUNG_EVERY_262144_MAX_POS_SERVING_MODEL_MUST_PASS: usize =
    192 * 1024;

fn ctx_tokens_from_env_default_256_8k_196608_matching_the_ctx196k_standard_and_the_g4moe_depth_rows(
) -> Vec<usize> {
    nv_ctx_tokens_env_with_k_suffix().unwrap_or_else(|| {
        vec![
            256,
            8 * 1024,
            DEPTH_196608_THE_CONFORMANCE_RUNG_EVERY_262144_MAX_POS_SERVING_MODEL_MUST_PASS,
        ]
    })
}

#[test]
#[ignore = "loads the ~10.4 GiB ig1 Qwen3.5-9B NVFP4 on wgpu (bf16 base via NV_QWEN35_DENSE_DIR); set NV_Q35_WGPU_SERVING_TEST=1 -- serving-shape decode ms/token vs KV depth on the 262144-max-pos 256/8k/196608 ladder: 8 full-attention layers read a KV window that is a pure function of pos while the 24 GDN linear-attention layers carry recurrent state with no depth dimension, so every state buffer is filled with finite synthetic bytes and pos set directly the way prefill would land it, then 64 eager decode_step calls are timed after a warmup plateau -- device argmax fed back, no host logits row, exactly what the chat_engine_wgpu greedy loop pays; run ONE depth per process via NV_CTX_TOKENS (the #96 residual race, same decoder module as the qwen38 wgpu instrument)"]
fn q35_dense_wgpu_serving_decode_ms_per_token_vs_context_depth_synthetic_state_fill_pos_is_the_only_depth_state(
) {
    if std::env::var("NV_Q35_WGPU_SERVING_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_Q35_WGPU_SERVING_TEST=1 to run this gate, or NV_MODELS_ALLOW_SKIP=1 to \
                 skip it on purpose; this family's served path is wgpu and it has no other depth \
                 instrument, so a silent ok here would hide exactly the gap task #206 exists to \
                 close"
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_Q35_WGPU_SERVING_TEST=1 to run");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let _ = THE_Q35_DENSE_WGPU_SERVING_DECODE_IS_EAGER_DECODE_STEP_DEVICE_ARGMAX_AND_THE_ENGINE_ARMS_NOTHING;
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_QWEN35_DENSE_DIR",
        &Q35_DENSE_WGPU_SERVES_THE_IG1_NVFP4_SNAPSHOT_BF16_BASE_IS_THE_ESCAPE_ARM,
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths =
        ctx_tokens_from_env_default_256_8k_196608_matching_the_ctx196k_standard_and_the_g4moe_depth_rows();
    let max_depth = depths.iter().copied().max().unwrap();

    let cfg = nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config");
    let max_seq = max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK;
    assert!(
        max_seq < cfg.max_position_embeddings,
        "ladder needs {max_seq} positions but max_position_embeddings is {}; a model declaring \
         that max_pos must serve depth {max_depth}, so a trip here is a config bug",
        cfg.max_position_embeddings
    );
    let n_full = cfg
        .layer_types
        .iter()
        .filter(|t| **t == nv_models::qwen3_5_moe::LayerType::FullAttention)
        .count();
    assert!(
        n_full > 0,
        "no full-attention layer sees KV depth on this config; a depth ladder over pure GDN \
         state would measure nothing depth-shaped"
    );
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut gpu = nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseWgpu::from_loader(
        cfg.clone(),
        &loader,
        max_seq,
    )
    .expect("build the 9B on the qwen3_5 dense wgpu decoder, the serving constructor's path");
    drop(loader);
    assert!(
        !gpu.mtp_active(),
        "a bare from_loader build must not carry the MTP head; serving attaches it only under \
         NV_Q3D_MTP=1 and this ladder's bare arm is the serving default"
    );
    eprintln!(
        "layers: {} total, {n_full} full-attention see depth, {} gdn carry no depth dimension",
        cfg.num_hidden_layers,
        cfg.num_hidden_layers - n_full
    );

    for &depth in &depths {
        gpu.reset().expect("reset between depth arms");
        let fill_start = std::time::Instant::now();
        gpu.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(depth)
            .unwrap_or_else(|e| panic!("synthetic state fill at depth {depth}: {e:#}"));
        ctx.poll_blocking()
            .expect("drain synthetic fill writes before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(
            gpu.current_pos(),
            depth,
            "synthetic fill must land at the requested depth"
        );

        let mut token = 2000u32;
        let (warmup_steps, mut step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                token = gpu
                    .decode_step(token)
                    .unwrap_or_else(|e| panic!("timed step at depth {depth}: {e:#}"));
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        let median = median_ms_of_timed_steps(&mut step_ms);
        eprintln!(
            "CTX-SCALING q35-dense-wgpu-serving-synthfill depth={depth} basis=eager_dense_wgpu_decode_step_synthetic_state_fill_the_serving_greedy_loop_is_decode_step_device_argmax median_ms_tok={median:.3} tok_s={:.1} fill_s={fill_s:.1} max_seq={max_seq} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
        ctx_timing_common::emit_nv_measure_v1_alongside_the_legacy_line_which_stays_for_one_release(
            ctx_timing_common::NvMeasureCtxLine {
                instrument: "ctx-scaling",
                model_dir: &dir,
                backend: "wgpu",
                device: ctx_timing_common::wgpu_device_name_for_the_nv_measure_device_basis_field(),
                tokens_primed_kv_depth_of_this_run_never_a_capacity: depth,
                steps: step_ms.len(),
                warmup: warmup_steps,
                value: median,
                unit: "ms/tok",
                basis: "eager_dense_wgpu_decode_step_synthetic_state_fill_the_serving_greedy_loop_is_decode_step_device_argmax",
            },
            &[
                ("fill_s", format!("{fill_s:.1}")),
                ("max_seq", format!("{max_seq}")),
            ],
        );
        print_wgpu_profile_share_table_then_reset(&format!("q35-dense-wgpu-serving depth={depth}"));
    }
}
