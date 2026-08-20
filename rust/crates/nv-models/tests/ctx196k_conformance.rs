#![cfg(any(feature = "cuda", feature = "wgpu"))]
#![allow(dead_code)]

mod common;
mod ctx_timing_common;

const DEPTH_196608_THE_CONFORMANCE_STANDARD_RUNG_EVERY_262144_MAX_POS_SERVING_MODEL_MUST_PASS:
    usize = 196608;
const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS: usize = 16;
const SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN_PREFILL_BLOCK: usize = 512;
const SYNTHETIC_FILL_CHUNK_256_RING_LEGAL_FOR_SLIDING_CACHES: usize = 256;

const Q38_196K_BOUND_IS_52_BECAUSE_TASK_110_MEASURED_41_7_MS_TOK_ON_THE_GRAPHED_DENSE_ENGINE_WITH_THE_GATED_SPLITK_DECODE_FIX_ARM_SO_THE_GATE_RUNS_THE_ARM:
    f64 = 52.0;
const Q36_196K_BOUND_IS_22_BECAUSE_MEASURED_17_45_MS_TOK_WITH_THE_GATED_SPLITK_ARM_AND_2255_WITHOUT_SO_THE_GATE_RUNS_THE_ARM:
    f64 = 22.0;
const G4_31B_196K_BOUND_IS_76_BECAUSE_TASK_106_MEASURED_60_8_MS_TOK_ON_THE_DEFAULT_EAGER_HYBRID_RING_ARM_THE_GRAPHED_POOL_GEOMETRY_CANNOT_HOLD_196K:
    f64 = 76.0;
const G4MOE_196K_BOUND_IS_70_BECAUSE_MEASURED_55_4_MS_TOK_WITH_THE_GATED_FLASH_DECODE_ARM_AND_1095_WITHOUT_SO_THE_GATE_RUNS_THE_ARM_PLUS_KV_RING_OR_THE_FLAT_SLIDING_CACHES_OOM:
    f64 = 70.0;
const LAGUNA_196K_BOUND_IS_33_BECAUSE_TASK_106_MEASURED_25_8_MS_TOK_ON_THE_DEFAULT_EAGER_ARM_NO_GATED_ARM_NEEDED:
    f64 = 33.0;

fn per_depth_extra_slots_warmup_plus_timed_plus_headroom() -> usize {
    ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
        + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
        + KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS
}

fn suite_gate_runs_or_skips_loudly(rung: &str) -> bool {
    if std::env::var("NV_CTX196K_CONFORMANCE").ok().as_deref() == Some("1") {
        return true;
    }
    if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
        panic!(
            "set NV_CTX196K_CONFORMANCE=1 to run the {rung} 196k conformance rung, or \
             NV_MODELS_ALLOW_SKIP=1 to skip it on purpose; a 196k gate that silently reports \
             ok would hide exactly the failure class task #106 exists to catch"
        );
    }
    eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_CTX196K_CONFORMANCE=1 to run {rung}");
    false
}

fn gate_sets_the_arm_env_because_the_gate_documents_what_serving_should_run(
    key: &str,
    why: &str,
) {
    if let Ok(v) = std::env::var(key) {
        assert_eq!(
            v, "1",
            "{key} is explicitly set to {v:?} but this conformance rung documents that \
             serving must run it: {why}"
        );
    }
    unsafe { std::env::set_var(key, "1") };
}

fn fixed_seed_small_nonzero_values_because_an_all_zero_cache_is_an_unrealistically_easy_softmax_and_collapses_moe_routing(
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

fn snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
    env_var: &str,
    repos: &[&str],
) -> std::path::PathBuf {
    if let Ok(d) = std::env::var(env_var) {
        return std::path::PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    for repo in repos {
        let base = std::path::PathBuf::from(&home)
            .join(".cache/huggingface/hub")
            .join(repo)
            .join("snapshots");
        let Ok(rd) = std::fs::read_dir(&base) else {
            continue;
        };
        let mut candidates: Vec<std::path::PathBuf> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.join("config.json").is_file()
                    && (p.join("model.safetensors").is_file()
                        || p.join("model.safetensors.index.json").is_file())
            })
            .collect();
        candidates.sort();
        if let Some(dir) = candidates.into_iter().next() {
            return dir;
        }
    }
    panic!("no snapshot with config.json + weights under HOME hub (tried {repos:?}); set {env_var}")
}

fn median_of_timed_steps(step_ms: &[f64]) -> f64 {
    let mut sorted = step_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn emit_conformance_line_then_assert_bound(
    model: &str,
    arm: &str,
    basis: &str,
    median_ms_tok: f64,
    bound_ms_tok: f64,
    fill_s: f64,
    warmup_steps: usize,
    timed_steps: usize,
) {
    assert!(
        median_ms_tok.is_finite() && median_ms_tok > 0.0,
        "median decode step time is not a positive number: {median_ms_tok}"
    );
    let verdict = if median_ms_tok <= bound_ms_tok {
        "PASS"
    } else {
        "FAIL"
    };
    eprintln!(
        "CONFORMANCE-196K model={model} depth={} arm={arm} basis={basis} median_ms_tok={median_ms_tok:.3} tok_s={:.1} bound_ms_tok={bound_ms_tok:.1} fill_s={fill_s:.1} warmup_steps={warmup_steps} steps={timed_steps} verdict={verdict}",
        DEPTH_196608_THE_CONFORMANCE_STANDARD_RUNG_EVERY_262144_MAX_POS_SERVING_MODEL_MUST_PASS,
        1000.0 / median_ms_tok
    );
    assert!(
        median_ms_tok <= bound_ms_tok,
        "{model} decoded at {median_ms_tok:.3} ms/tok at the 196k rung, over the conformance \
         bound {bound_ms_tok:.1} whose constant names today's measured reality on arm {arm}; \
         a trip here is a serving regression to diagnose, not a context cap to accept"
    );
}

#[cfg(feature = "cuda")]
fn bf16_kv_chunk_templates(
    device: &candle_core::Device,
    chunk: usize,
    n_kv: usize,
    hd: usize,
) -> (candle_core::Tensor, candle_core::Tensor) {
    use candle_core::{DType, Tensor};
    let vals = fixed_seed_small_nonzero_values_because_an_all_zero_cache_is_an_unrealistically_easy_softmax_and_collapses_moe_routing(chunk * n_kv * hd);
    let k = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), device)
        .expect("k template")
        .to_dtype(DType::BF16)
        .expect("k bf16");
    let v = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), device)
        .expect("v template")
        .to_dtype(DType::BF16)
        .expect("v bf16");
    (k, v)
}

#[cfg(feature = "cuda")]
fn synth_fill_graphed_qwen_engine_to_depth(
    eng: &mut nv_models::graph_engine::GraphedQwen3Moe,
    k_template: &candle_core::Tensor,
    v_template: &candle_core::Tensor,
    chunk: usize,
    depth: usize,
) -> f64 {
    let fill_start = std::time::Instant::now();
    let mut pos = 0usize;
    while pos < depth {
        let n = chunk.min(depth - pos);
        if n == chunk {
            eng.prime_kv_depth_synthetically_for_ctx_timing_decode_reads_cache_size_not_values(
                k_template, v_template,
            )
            .unwrap_or_else(|e| panic!("synthetic fill chunk at pos {pos}: {e:#}"));
        } else {
            let kn = k_template.narrow(1, 0, n).expect("k tail");
            let vn = v_template.narrow(1, 0, n).expect("v tail");
            eng.prime_kv_depth_synthetically_for_ctx_timing_decode_reads_cache_size_not_values(
                &kn, &vn,
            )
            .unwrap_or_else(|e| panic!("synthetic fill tail at pos {pos}: {e:#}"));
        }
        pos += n;
    }
    eng.synchronize().expect("sync before stopping the fill clock");
    assert_eq!(
        eng.current_pos(),
        depth,
        "synthetic fill must advance current_pos exactly the way prefill would"
    );
    fill_start.elapsed().as_secs_f64()
}

#[cfg(feature = "cuda")]
fn timed_graphed_free_running_decode(
    eng: &mut nv_models::graph_engine::GraphedQwen3Moe,
    depth: usize,
) -> (usize, Vec<f64>) {
    let mut cur = 2000u32;
    let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
        || {
            eng.forward_decode(cur)
                .unwrap_or_else(|e| panic!("serving decode at depth {depth}: {e:#}"));
            let r = eng.logits_host().expect("logits to host");
            cur = common::argmax(&r);
        },
        TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
    );
    assert!(
        eng.capture_active(),
        "the serving basis of this gate is the CAPTURED graphed decode and the engine fell \
         back to uncaptured at depth {depth}; the engine printed its blocker to stderr just \
         above -- diagnose that blocker, do not relabel the number as serving-class"
    );
    (warmup_steps, step_ms)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 dense arm GRAPHED; set NV_CTX196K_CONFORMANCE=1 -- the task #106 executable standard at 196608: synthetic fp8 KV fill, captured graphed free-running decode, median ms/tok under the bound constant; the gate itself sets NV_Q36_GRAPHED_DECODE_FIX=1 because that splitk arm is what serving should run; run in its own process"]
fn ctx196k_qwen38_27b_cuda_graphed_dense_serving_synthfill_meets_bound() {
    use candle_core::Device;
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
    use nv_weights::{QuantizationConfig, WeightLoader};
    if !suite_gate_runs_or_skips_loudly("qwen38-27b") {
        return;
    }
    gate_sets_the_arm_env_because_the_gate_documents_what_serving_should_run(
        "NV_Q36_GRAPHED_DECODE_FIX",
        "routes decode_attention_fp8 to the splitk flash_decode_fused_fp8kv kernel; the \
         measured reality behind the bound constant is with this arm on (current numbers: \
         perf/runs.jsonl)",
    );
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = common::qwen38_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, &device)
        .expect("build Qwen3.8-27B on the dense arm for the graphed conformance rung");
    drop(weights);
    assert!(
        model.is_dense(),
        "from_loader_dense_quantized must yield the dense arm"
    );

    let depth =
        DEPTH_196608_THE_CONFORMANCE_STANDARD_RUNG_EVERY_262144_MAX_POS_SERVING_MODEL_MUST_PASS;
    let max_pos = model.config().max_position_embeddings;
    let extra = per_depth_extra_slots_warmup_plus_timed_plus_headroom();
    assert!(
        depth + extra < max_pos,
        "depth {depth} + decode steps exceeds max_position_embeddings {max_pos}; a model \
         declaring {max_pos} must serve this depth, so a trip here is a config bug"
    );
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let mut eng = GraphedQwen3Moe::new(model, &device, depth + extra)
        .unwrap_or_else(|e| panic!("GraphedQwen3Moe with {} kv slots: {e:#}", depth + extra));

    let chunk = SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN_PREFILL_BLOCK;
    let (k_template, v_template) = bf16_kv_chunk_templates(&device, chunk, n_kv, hd);
    eng.set_moe_dispatch(None);
    eng.reset().expect("engine reset");
    let fill_s = synth_fill_graphed_qwen_engine_to_depth(&mut eng, &k_template, &v_template, chunk, depth);
    eng.install_grouped_moe()
        .expect("dense branch of install_grouped_moe arms capture without a dispatch");
    assert!(
        eng.moe_dispatch_ref().is_none(),
        "the dense arm must not carry a MoE dispatch"
    );

    let (warmup_steps, step_ms) = timed_graphed_free_running_decode(&mut eng, depth);
    let median = median_of_timed_steps(&step_ms);
    emit_conformance_line_then_assert_bound(
        "qwen38-27b",
        "graphed_dense+NV_Q36_GRAPHED_DECODE_FIX=1_splitk_flash",
        "captured_graphed_free_running_decode_argmax_feed_includes_logits_host_synthetic_kv_fill",
        median,
        Q38_196K_BOUND_IS_52_BECAUSE_TASK_110_MEASURED_41_7_MS_TOK_ON_THE_GRAPHED_DENSE_ENGINE_WITH_THE_GATED_SPLITK_DECODE_FIX_ARM_SO_THE_GATE_RUNS_THE_ARM,
        fill_s,
        warmup_steps,
        step_ms.len(),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B NVFP4 GRAPHED grouped-MoE; set NV_CTX196K_CONFORMANCE=1 -- the task #106 executable standard at 196608: synthetic fp8 KV fill, captured graphed free-running decode, median ms/tok under the bound constant; the gate itself sets NV_Q36_GRAPHED_DECODE_FIX=1 because without the splitk arm the same rung misses the bound by an order of magnitude (current numbers: perf/runs.jsonl); run in its own process"]
fn ctx196k_qwen36_35b_cuda_graphed_grouped_moe_serving_synthfill_meets_bound_with_the_splitk_arm() {
    use candle_core::Device;
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
    use nv_weights::{QuantizationConfig, WeightLoader};
    if !suite_gate_runs_or_skips_loudly("qwen36-35b") {
        return;
    }
    gate_sets_the_arm_env_because_the_gate_documents_what_serving_should_run(
        "NV_Q36_GRAPHED_DECODE_FIX",
        "routes decode_attention_fp8 to the splitk flash_decode_fused_fp8kv kernel; the \
         splitk arm outruns the unfixed route by two orders of magnitude at this depth \
         (current numbers: perf/runs.jsonl) -- the gate documents that serving must run it",
    );
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = common::qwen36_nvfp4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);

    let depth =
        DEPTH_196608_THE_CONFORMANCE_STANDARD_RUNG_EVERY_262144_MAX_POS_SERVING_MODEL_MUST_PASS;
    let max_pos = model.config().max_position_embeddings;
    let extra = per_depth_extra_slots_warmup_plus_timed_plus_headroom();
    assert!(
        depth + extra < max_pos,
        "depth {depth} + decode steps exceeds max_position_embeddings {max_pos}; a model \
         declaring {max_pos} must serve this depth, so a trip here is a config bug"
    );
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let mut eng = GraphedQwen3Moe::new(model, &device, depth + extra)
        .unwrap_or_else(|e| panic!("GraphedQwen3Moe with {} kv slots: {e:#}", depth + extra));

    let chunk = SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN_PREFILL_BLOCK;
    let (k_template, v_template) = bf16_kv_chunk_templates(&device, chunk, n_kv, hd);
    eng.set_moe_dispatch(None);
    eng.reset().expect("engine reset");
    let fill_s = synth_fill_graphed_qwen_engine_to_depth(&mut eng, &k_template, &v_template, chunk, depth);
    eng.install_grouped_moe().expect("grouped moe install");

    let (warmup_steps, step_ms) = timed_graphed_free_running_decode(&mut eng, depth);
    let median = median_of_timed_steps(&step_ms);
    emit_conformance_line_then_assert_bound(
        "qwen36-35b",
        "graphed_grouped_moe+NV_Q36_GRAPHED_DECODE_FIX=1_splitk_flash",
        "captured_graphed_free_running_decode_argmax_feed_includes_logits_host_synthetic_kv_fill",
        median,
        Q36_196K_BOUND_IS_22_BECAUSE_MEASURED_17_45_MS_TOK_WITH_THE_GATED_SPLITK_ARM_AND_2255_WITHOUT_SO_THE_GATE_RUNS_THE_ARM,
        fill_s,
        warmup_steps,
        step_ms.len(),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "loads the 31B; set NV_CTX196K_CONFORMANCE=1 -- the task #106 executable standard at 196608 on the arm that reaches 196k: eager paged fp8 hybrid-ring forward_decode_batched at defaults, synthetic cache fill; the graphed Gemma4BatchGraphFamily arm cannot hold a lanes==0 pool at this depth (pool geometry, not capture shape); run in its own process"]
fn ctx196k_gemma4_31b_cuda_serving_eager_hybrid_ring_synthfill_meets_bound() {
    use candle_core::{DType, Device, Tensor};
    use nv_models::gemma4::{Gemma4, Gemma4Cache, Gemma4Config};
    use nv_models::paged_fp8::{PagedGemma4Cache, PagedKvFp8Pool, PagedPoolConfig};
    use nv_weights::{QuantizationConfig, WeightLoader};
    use std::sync::{Arc, Mutex};
    if !suite_gate_runs_or_skips_loudly("gemma4-31b") {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = common::gemma4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model =
        Arc::new(Gemma4::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model"));
    drop(weights);

    let depth =
        DEPTH_196608_THE_CONFORMANCE_STANDARD_RUNG_EVERY_262144_MAX_POS_SERVING_MODEL_MUST_PASS;
    let max_pos = model.config().max_position_embeddings;
    let vocab = model.config().vocab_size;
    let bs = 16usize;
    let total_slots = depth + per_depth_extra_slots_warmup_plus_timed_plus_headroom();
    assert!(
        total_slots < max_pos,
        "depth {depth} + decode steps exceeds max_position_embeddings {max_pos}; a model \
         declaring {max_pos} must serve this depth, so a trip here is a config bug"
    );
    let n_layers = model.config().num_hidden_layers;
    let chunk = SYNTHETIC_FILL_CHUNK_256_RING_LEGAL_FOR_SLIDING_CACHES;
    let templates: Vec<(Tensor, Tensor)> = (0..n_layers)
        .map(|li| {
            let kind = model.config().layer_kind(li);
            let hd = model.config().head_dim_for(kind);
            let n_kv = model.config().num_kv_heads_for(kind);
            let vals = fixed_seed_small_nonzero_values_because_an_all_zero_cache_is_an_unrealistically_easy_softmax_and_collapses_moe_routing(chunk * n_kv * hd);
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

    let seq_blocks = total_slots.div_ceil(bs);
    let hybrid_cfg = PagedPoolConfig::from_gemma4_hybrid(model.config(), seq_blocks, bs, 1);
    let pool = Arc::new(Mutex::new(
        PagedKvFp8Pool::new(hybrid_cfg, &device).expect("hybrid ring paged fp8 pool"),
    ));
    let mut cache = PagedGemma4Cache::new(pool.clone(), &device).expect("cache");
    let table: Vec<u32> = (0..seq_blocks as u32).collect();
    cache.set_block_table(&table).expect("block table");

    let fill_start = std::time::Instant::now();
    let mut pos = 0usize;
    while pos < depth {
        let n = chunk.min(depth - pos);
        cache
            .prepare_for_decode(pos, pos + n)
            .expect("prepare_for_decode");
        for (li, (k, v)) in templates.iter().enumerate() {
            if n == chunk {
                cache.write_at(li, k, v).expect("write_at");
            } else {
                let kn = k.narrow(1, 0, n).expect("k tail");
                let vn = v.narrow(1, 0, n).expect("v tail");
                cache.write_at(li, &kn, &vn).expect("write_at tail");
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
        "synthetic fill must leave the paged cache at the requested depth"
    );

    let mut tok = 2000u32;
    let mut p = depth;
    let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
        || {
            let mut caches: Vec<&mut PagedGemma4Cache> = vec![&mut cache];
            let logits = model
                .forward_decode_batched(&[tok], &[p], &mut caches)
                .unwrap_or_else(|e| panic!("paged batched decode at depth {p}: {e:#}"));
            let v: Vec<f32> = logits
                .to_dtype(DType::F32)
                .expect("f32")
                .flatten_all()
                .expect("flatten")
                .to_vec1()
                .expect("host");
            tok = common::argmax(&v[0..vocab]);
            p += 1;
        },
        TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
    );
    let median = median_of_timed_steps(&step_ms);
    emit_conformance_line_then_assert_bound(
        "gemma4-31b",
        "default_eager_paged_fp8_hybrid_ring_forward_decode_batched",
        "eager_serving_decode_host_row_included_synthetic_kv_fill",
        median,
        G4_31B_196K_BOUND_IS_76_BECAUSE_TASK_106_MEASURED_60_8_MS_TOK_ON_THE_DEFAULT_EAGER_HYBRID_RING_ARM_THE_GRAPHED_POOL_GEOMETRY_CANNOT_HOLD_196K,
        fill_s,
        warmup_steps,
        step_ms.len(),
    );
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "loads the 26B-A4B MoE on wgpu; set NV_CTX196K_CONFORMANCE=1 -- the task #106 executable standard at 196608: synthetic state-buffer fill with pos set directly, timed decode_step, median ms/tok under the bound constant; the gate itself sets NV_G4MOE_KV_RING=1 (flat sliding caches OOM at this depth) and NV_G4MOE_FLASH_DECODE=1 (the flash arm outruns the serial arm by over an order of magnitude at this depth; current numbers: perf/runs.jsonl) because those arms are what serving should run; run in its own process"]
fn ctx196k_gemma4_26b_a4b_wgpu_ring_flash_decode_synthfill_meets_bound_with_the_gated_arms() {
    if !suite_gate_runs_or_skips_loudly("gemma4-26b-a4b") {
        return;
    }
    gate_sets_the_arm_env_because_the_gate_documents_what_serving_should_run(
        "NV_G4MOE_KV_RING",
        "the flat sliding caches are ~42 GiB bf16 KV at 196k and OOM the card; the ring is \
         bit-identical to full depth far past the wraparound",
    );
    gate_sets_the_arm_env_because_the_gate_documents_what_serving_should_run(
        "NV_G4MOE_FLASH_DECODE",
        "the flash full-attention decode arm outruns the serial arm by over an order of \
         magnitude at 196k (current numbers: perf/runs.jsonl) -- the gate documents that \
         serving must run it",
    );
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_G4MOE_SNAPSHOT",
        &["models--google--gemma-4-26B-A4B-it"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated conformance rung must never silently skip");
    eprintln!("adapter: {}", ctx.summary());

    let config =
        nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json"))
            .expect("config.json");
    let depth =
        DEPTH_196608_THE_CONFORMANCE_STANDARD_RUNG_EVERY_262144_MAX_POS_SERVING_MODEL_MUST_PASS;
    let max_seq = depth + per_depth_extra_slots_warmup_plus_timed_plus_headroom();
    assert!(
        max_seq <= config.base.max_position_embeddings,
        "rung needs {max_seq} positions but max_position_embeddings is {}; a model declaring \
         262144 must serve this depth, so a trip here is a config bug",
        config.base.max_position_embeddings
    );
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut m = nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu::from_loader(config, &loader, max_seq)
        .expect("build from loader");
    drop(loader);
    assert!(
        nv_models::gemma4_moe_wgpu::sliding_kv_ring_enabled(),
        "the gate set NV_G4MOE_KV_RING=1 but the build did not take the ring"
    );
    assert!(
        nv_models::gemma4_moe_wgpu::flash_decode_enabled(),
        "the gate set NV_G4MOE_FLASH_DECODE=1 but the flash arm is off"
    );

    let fill_start = std::time::Instant::now();
    m.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(depth)
        .unwrap_or_else(|e| panic!("synthetic state fill at depth {depth}: {e:#}"));
    ctx.poll_blocking()
        .expect("drain synthetic fill writes before stopping the fill clock");
    let fill_s = fill_start.elapsed().as_secs_f64();
    assert_eq!(
        m.current_pos(),
        depth,
        "synthetic fill must land at the requested depth"
    );

    let mut token = 2000u32;
    let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
        || {
            token = m
                .decode_step(token)
                .unwrap_or_else(|e| panic!("timed step at depth {depth}: {e:#}"));
        },
        TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
    );
    let median = median_of_timed_steps(&step_ms);
    emit_conformance_line_then_assert_bound(
        "gemma4-26b-a4b",
        "wgpu+NV_G4MOE_KV_RING=1+NV_G4MOE_FLASH_DECODE=1",
        "wgpu_decode_step_synthetic_state_fill_pos_is_the_only_depth_state",
        median,
        G4MOE_196K_BOUND_IS_70_BECAUSE_MEASURED_55_4_MS_TOK_WITH_THE_GATED_FLASH_DECODE_ARM_AND_1095_WITHOUT_SO_THE_GATE_RUNS_THE_ARM_PLUS_KV_RING_OR_THE_FLAT_SLIDING_CACHES_OOM,
        fill_s,
        warmup_steps,
        step_ms.len(),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "loads the Laguna-XS-2.1 NVFP4; set NV_CTX196K_CONFORMANCE=1 -- the task #106 executable standard at 196608 on the default eager cuda arm: only the attention-KV rows are synthetically deep (ring-legal 256 chunks), timed forward_with_cache decode, median ms/tok under the bound constant; run in its own process"]
fn ctx196k_laguna_xs21_cuda_eager_synthfill_meets_bound() {
    use candle_core::{DType, Device, Tensor};
    use nv_models::gemma4::Gemma4Cache;
    use nv_models::laguna::{Laguna, LagunaConfig};
    use nv_weights::{QuantizationConfig, WeightLoader};
    if !suite_gate_runs_or_skips_loudly("laguna-xs21") {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_LAGUNA_DIR",
        &["models--poolside--Laguna-XS-2.1-NVFP4"],
    );
    let device = Device::new_cuda(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = LagunaConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Laguna::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);

    let depth =
        DEPTH_196608_THE_CONFORMANCE_STANDARD_RUNG_EVERY_262144_MAX_POS_SERVING_MODEL_MUST_PASS;
    let max_pos = model.config().max_position_embeddings;
    let extra = per_depth_extra_slots_warmup_plus_timed_plus_headroom();
    assert!(
        depth + extra < max_pos,
        "depth {depth} + decode steps exceeds max_position_embeddings {max_pos}; a model \
         declaring {max_pos} must serve this depth, so a trip here is a config bug"
    );
    let n_layers = model.config().num_hidden_layers;
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let chunk = SYNTHETIC_FILL_CHUNK_256_RING_LEGAL_FOR_SLIDING_CACHES;
    let vals = fixed_seed_small_nonzero_values_because_an_all_zero_cache_is_an_unrealistically_easy_softmax_and_collapses_moe_routing(chunk * n_kv * hd);
    let k_template = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), &device)
        .expect("k template")
        .to_dtype(DType::BF16)
        .expect("k bf16");
    let v_template = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), &device)
        .expect("v template")
        .to_dtype(DType::BF16)
        .expect("v bf16");

    let mut cache = model.new_kv_cache(depth + extra).expect("kv cache");
    let fill_start = std::time::Instant::now();
    let mut pos = 0usize;
    while pos < depth {
        let n = chunk.min(depth - pos);
        cache
            .prepare_for_decode(pos, pos + n)
            .expect("prepare_for_decode");
        for li in 0..n_layers {
            if n == chunk {
                cache
                    .write_at(li, &k_template, &v_template)
                    .expect("write_at");
            } else {
                let kn = k_template.narrow(1, 0, n).expect("k tail");
                let vn = v_template.narrow(1, 0, n).expect("v tail");
                cache.write_at(li, &kn, &vn).expect("write_at tail");
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
        "synthetic fill must leave the cache at the requested depth"
    );

    let mut p = depth;
    let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
        || {
            let tokens = Tensor::from_vec(
                vec![2000u32 + (p as u32 % 30000)],
                (1usize, 1usize),
                &device,
            )
            .expect("token");
            let positions = Tensor::from_vec(vec![p as i32], 1usize, &device).expect("position");
            let logits = model
                .forward_with_cache(&tokens, &positions, &mut cache)
                .unwrap_or_else(|e| panic!("decode at depth {p}: {e:#}"));
            logits
                .to_dtype(DType::F32)
                .expect("f32")
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("sync to host");
            p += 1;
        },
        TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
    );
    let median = median_of_timed_steps(&step_ms);
    emit_conformance_line_then_assert_bound(
        "laguna-xs21",
        "default_eager_cuda_forward_with_cache",
        "eager_decode_host_row_included_synthetic_kv_fill_attention_rows_only",
        median,
        LAGUNA_196K_BOUND_IS_33_BECAUSE_TASK_106_MEASURED_25_8_MS_TOK_ON_THE_DEFAULT_EAGER_ARM_NO_GATED_ARM_NEEDED,
        fill_s,
        warmup_steps,
        step_ms.len(),
    );
}

#[test]
#[ignore = "RED LIST, never run as a timing rung: Qwen3.5-9B declares max_position_embeddings 262144 but has no 196k-capable serving path -- the only cuda arm is the eager dense forward_with_cache, which decodes far past any serving bound at depth (qwen35_9b_cuda_decode_ms_per_token_vs_context_depth_eager_dense_chunked_prefill_primed; current numbers: perf/runs.jsonl), there is no graphed engine and no splitk decode route for this family; running this test documents the blocker by panicking, which is the point: a model on the red list is a bug to fix, not a silent skip"]
fn ctx196k_qwen35_9b_red_list_no_196k_capable_serving_path() {
    panic!(
        "Qwen3.5-9B is on the 196k conformance RED LIST: max_position_embeddings 262144 with \
         no serving-class 196k path. Blocker: the family has only the eager dense cuda arm \
         (from_loader_dense_quantized + forward_with_cache), which measures orders of \
         magnitude over the serving bar at depth (current numbers: perf/runs.jsonl), \
         and no GraphedQwen3Moe-class captured decode or \
         splitk flash decode route is wired for it. Fix is to give the 9B the graphed dense \
         serving engine the 27B already has, then move this rung off the red list by giving \
         it a bound constant and a synthfill body like the other rungs in this file."
    );
}
