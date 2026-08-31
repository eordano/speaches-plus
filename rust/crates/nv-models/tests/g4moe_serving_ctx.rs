#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use common::percentile_of_sorted;
use common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::Gemma4Cache;
use nv_models::gemma4_moe::{
    nv_g4moe_cuda_nvfp4_opt_in_because_post_training_quant_failed_the_ppl_band,
    nv_g4moe_expert_bt_gemm_opt_in_because_the_route_change_is_banded_not_bitwise, Gemma4Moe,
    Gemma4MoeConfig,
};
use nv_weights::WeightLoader;
use std::time::Instant;

mod ctx_timing_common;

pub const THE_G4MOE_CUDA_SERVING_DECODE_IS_EAGER_FORWARD_WITH_CACHE_OVER_THE_BF16_SLIDING_RING_KV_AND_BUILDS_NO_GRAPH: &str =
    "rust/src/oapi/chat_engine/gemma4_moe_loop.rs::run_sampling_gemma4_moe allocates \
     model.new_kv_cache(cache_len) -- the bf16 Gemma4KvCache whose sliding layers cap at \
     sliding_window+SLIDING_COMPACT_SLACK while the five full-attention layers stay contiguous \
     at full depth -- chunk-prefills it at PREFILL_CHUNK=256 through forward_with_cache, and \
     then decodes every token through the same eager forward_with_cache one [1,1] step at a \
     time, copying the full logits row to the host and sampling there. \
     chat_engine/build.rs states the graph fact directly: 'Gemma4Moe and Qwen3 build no \
     graph'. There is no graphed or captured arm for this family on cuda, so the eager step IS \
     the serving shape, and this ladder times it with the host logits row and host argmax \
     included exactly as the serving loop pays for them.";

const G4MOE_HOME_HUB_REPO_MATCHING_THE_WGPU_AND_PPL_ROWS: &str =
    "models--google--gemma-4-26B-A4B-it";

const SYNTHETIC_FILL_CHUNK_256_THE_SLIDING_RING_CAP_IS_WINDOW_PLUS_COMPACT_SLACK_256_SO_A_LARGER_APPEND_TRIPS_THE_CAPACITY_BAIL:
    usize = 256;
const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS: usize = 16;
const DEPTH_196608_MATCHES_THE_WGPU_G4MOE_DEPTH_ROWS_SO_CUDA_AND_WGPU_COLUMNS_ALIGN: usize =
    192 * 1024;

fn ctx_tokens_from_env_default_256_8k_196608_the_wgpu_comparable_deep_rung() -> Vec<usize> {
    ctx_timing_common::nv_ctx_tokens_env_with_k_suffix().unwrap_or_else(|| {
        vec![
            256,
            8 * 1024,
            DEPTH_196608_MATCHES_THE_WGPU_G4MOE_DEPTH_ROWS_SO_CUDA_AND_WGPU_COLUMNS_ALIGN,
        ]
    })
}

fn fixed_seed_small_nonzero_values_so_attention_reads_are_realistic_and_moe_routing_is_not_collapsed_toward_one_expert(
    len: usize,
) -> Vec<f32> {
    let mut state = 0x9e3779b97f4a7c15u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 0.5
        })
        .collect()
}

fn arm_label_from_the_accessors_the_serving_dispatch_and_cache_actually_read(
    model: &Gemma4Moe,
) -> String {
    fn on_off(on: bool) -> &'static str {
        if on {
            "on"
        } else {
            "off"
        }
    }
    format!(
        "NV_G4MOE_CUDA_NVFP4={},NV_G4MOE_EXPERT_BT_GEMM={},NV_KV_NO_SLIDING={}",
        on_off(model.experts_nvfp4_resident_layer_count() > 0),
        on_off(nv_g4moe_expert_bt_gemm_opt_in_because_the_route_change_is_banded_not_bitwise()),
        on_off(nv_config::env_arm_present_at_all_reread_every_call("NV_KV_NO_SLIDING")),
    )
}

#[test]
#[ignore = "loads the ~38 GB google/gemma-4-26B-A4B-it bf16 checkpoint on cuda; set NV_G4MOE_SERVING_TEST=1 -- serving-shape decode ms/token vs KV depth on the 262144-max-pos 256/8k/196608 ladder (196608 matches the wgpu depth rows): the bf16 Gemma4KvCache is filled synthetically through prepare_for_decode/write_at/advance in ring-legal 256 chunks and current_len advanced the way prefill would (decode cost reads cache SIZE, not values; the fill is fixed-seed nonzero so routing does not collapse), then 64 eager forward_with_cache steps are timed after a warmup plateau with the host logits row and host argmax included exactly as run_sampling_gemma4_moe pays for them -- this family's serving decode builds no graph, so eager IS the serving shape, and every depth allocates a fresh cache, so a multi-depth NV_CTX_TOKENS run shares no state across rungs"]
fn g4moe_cuda_serving_decode_ms_per_token_vs_context_depth_synthetic_cache_fill_eager_bf16_ring_kv_the_serving_decode_builds_no_graph(
) {
    if std::env::var("NV_G4MOE_SERVING_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_G4MOE_SERVING_TEST=1 to run this gate, or NV_MODELS_ALLOW_SKIP=1 to \
                 skip it on purpose; a 196k serving gate that silently reports ok would hide \
                 exactly the failure class task #106 exists to catch"
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_G4MOE_SERVING_TEST=1 to run");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let _ = THE_G4MOE_CUDA_SERVING_DECODE_IS_EAGER_FORWARD_WITH_CACHE_OVER_THE_BF16_SLIDING_RING_KV_AND_BUILDS_NO_GRAPH;
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_G4MOE_SNAPSHOT",
        &[G4MOE_HOME_HUB_REPO_MATCHING_THE_WGPU_AND_PPL_ROWS],
    );
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let config =
        Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("parse moe config");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Gemma4Moe::from_loader(config, &weights, &device).expect(
        "build gemma4-26B-A4B via from_loader, the serving constructor whose env selects the \
         expert residency",
    );
    drop(weights);
    let nvfp4_layers = model.experts_nvfp4_resident_layer_count();
    if nv_g4moe_cuda_nvfp4_opt_in_because_post_training_quant_failed_the_ppl_band() {
        assert_eq!(
            nvfp4_layers,
            model.config().base.num_hidden_layers,
            "NV_G4MOE_CUDA_NVFP4=1 must leave every layer's experts nvfp4-resident or the arm \
             label lies about what was timed"
        );
    } else {
        assert_eq!(
            nvfp4_layers, 0,
            "serving defaults must keep every expert stack bf16; a stray nvfp4-resident layer \
             here means the ladder's bare arm is not the serving default"
        );
    }

    let depths = ctx_tokens_from_env_default_256_8k_196608_the_wgpu_comparable_deep_rung();
    let max_depth = depths.iter().copied().max().unwrap();
    let max_pos = model.config().base.max_position_embeddings;
    let per_depth_extra_slots =
        ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
            + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
            + KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS;
    assert!(
        max_depth + per_depth_extra_slots < max_pos,
        "depth {max_depth} + decode steps exceeds max_position_embeddings {max_pos}; a model \
         declaring {max_pos} must serve this depth, so a trip here is a config bug"
    );

    let n_layers = model.config().base.num_hidden_layers;
    let vocab = model.config().base.vocab_size;
    let chunk =
        SYNTHETIC_FILL_CHUNK_256_THE_SLIDING_RING_CAP_IS_WINDOW_PLUS_COMPACT_SLACK_256_SO_A_LARGER_APPEND_TRIPS_THE_CAPACITY_BAIL;
    let templates: Vec<(Tensor, Tensor)> = (0..n_layers)
        .map(|li| {
            let kind = model.config().base.layer_kind(li);
            let hd = model.config().base.head_dim_for(kind);
            let n_kv = model.config().base.num_kv_heads_for(kind);
            let vals = fixed_seed_small_nonzero_values_so_attention_reads_are_realistic_and_moe_routing_is_not_collapsed_toward_one_expert(
                chunk * n_kv * hd,
            );
            let k = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), &device)
                .expect("k template")
                .to_dtype(DType::BF16)
                .expect("k bf16");
            let v = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), &device)
                .expect("v template")
                .to_dtype(DType::BF16)
                .expect("v bf16");
            (k, v)
        })
        .collect();

    for &depth in &depths {
        let mut cache = model
            .new_kv_cache(depth + per_depth_extra_slots)
            .expect("bf16 sliding-ring kv cache, the one run_sampling_gemma4_moe allocates");

        let fill_start = Instant::now();
        let mut pos = 0usize;
        while pos < depth {
            let n = chunk.min(depth - pos);
            cache
                .prepare_for_decode(pos, pos + n)
                .expect("prepare_for_decode");
            for (li, (k, v)) in templates.iter().enumerate() {
                if n == chunk {
                    cache
                        .write_at(li, k, v)
                        .unwrap_or_else(|e| panic!("synthetic fill chunk at pos {pos}: {e:#}"));
                } else {
                    let kn = k.narrow(1, 0, n).expect("k tail");
                    let vn = v.narrow(1, 0, n).expect("v tail");
                    cache
                        .write_at(li, &kn, &vn)
                        .unwrap_or_else(|e| panic!("synthetic fill tail at pos {pos}: {e:#}"));
                }
            }
            cache.advance(n);
            pos += n;
        }
        device
            .synchronize()
            .expect("sync before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(
            cache.current_len(),
            depth,
            "synthetic fill must advance current_len exactly the way prefill would"
        );

        let mut tok = 2000u32;
        let mut step_pos = depth;
        let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                let tokens = Tensor::from_vec(vec![tok], (1usize, 1usize), &device)
                    .expect("token");
                let positions =
                    Tensor::from_vec(vec![step_pos as i32], 1usize, &device).expect("position");
                let logits = model
                    .forward_with_cache(&tokens, &positions, &mut cache)
                    .unwrap_or_else(|e| panic!("serving decode at depth {step_pos}: {e:#}"));
                let row: Vec<f32> = logits
                    .to_dtype(DType::F32)
                    .expect("f32")
                    .flatten_all()
                    .expect("flatten")
                    .to_vec1()
                    .expect("host row");
                assert_eq!(
                    row.len(),
                    vocab,
                    "a [1,1] serving decode step must hand back one full logits row"
                );
                tok = argmax(&row);
                step_pos += 1;
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );

        let mut sorted = step_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = percentile_of_sorted(&sorted, 0.5);
        let p10 = percentile_of_sorted(&sorted, 0.1);
        let p90 = percentile_of_sorted(&sorted, 0.9);
        assert!(
            median.is_finite() && median > 0.0,
            "median decode step time is not a positive number: {median}"
        );
        let arm = arm_label_from_the_accessors_the_serving_dispatch_and_cache_actually_read(&model);
        eprintln!(
            "CTX-SCALING g4moe-cuda-serving-synthfill depth={depth} basis=eager_forward_with_cache_the_serving_decode_builds_no_graph_bf16_ring_kv_host_row_argmax_feed_synthetic_kv_fill arm={arm} nvfp4_layers={nvfp4_layers} median_ms_tok={median:.3} p10={p10:.3} p90={p90:.3} tok_s={:.1} fill_s={fill_s:.1} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
        ctx_timing_common::emit_nv_measure_v1_alongside_the_legacy_line_which_stays_for_one_release(
            ctx_timing_common::NvMeasureCtxLine {
                instrument: "ctx-scaling",
                model_dir: &dir,
                backend: "cuda",
                device: ctx_timing_common::cuda_device_name_for_the_nv_measure_device_basis_field(),
                tokens_primed_kv_depth_of_this_run_never_a_capacity: depth,
                steps: step_ms.len(),
                warmup: warmup_steps,
                value: median,
                unit: "ms/tok",
                basis: "eager_forward_with_cache_the_serving_decode_builds_no_graph_bf16_ring_kv_host_row_argmax_feed_synthetic_kv_fill",
            },
            &[
                ("arm", arm),
                ("fill_s", format!("{fill_s:.1}")),
            ],
        );
    }
}
