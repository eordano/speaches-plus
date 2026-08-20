#![cfg(feature = "cuda")]

mod common;
use common::qwen38_snapshot_dir_env_override_then_home_hub;
use candle_core::{DType, Device, Tensor};
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use std::time::Instant;

mod ctx_timing_common;

const SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN35_DENSE_PREFILL_BLOCK: usize = 512;
const GRAPHED_TIMED_STEPS_32_A_MEDIAN_NOT_A_LADDER: usize = 32;
const ATTRIBUTED_EAGER_STEPS_12_EACH_PRINTS_A_PROF_DECODE_SPLIT: usize = 12;
const KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS: usize = 16;

fn depth_from_env_default_256() -> usize {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => {
            let s = v.trim();
            let (num, mult) = match s.strip_suffix('k') {
                Some(n) => (n, 1024usize),
                None => (s, 1usize),
            };
            num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
        }
        Err(_) => 256,
    }
}

fn fixed_seed_values_because_an_all_zero_cache_is_an_unrealistically_easy_softmax(
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

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 dense arm; set NV_QWEN38_SERVING_TEST=1 -- the decode-gap attribution probe: phase A captures the graphed m=1 decode (asserts capture_active, prints the captured node count and a graphed median), phase B flips NV_GRAPH_OFF=1 + NV_PROF_DECODE=1 in-process so the same engine decodes uncaptured while every step prints a [prof-decode] per-bucket wall split (qkv gemvs, qk-norm+rope glue, kv fp8 store, splitk attention core, gate+o_proj, gdn chain, dense mlp, norms+residual, final norm, lm_head); the eager TOTAL exceeds the graphed median by the sync+launch overhead, so read the split as attribution, not as speed; one depth per process via NV_CTX_TOKENS"]
fn qwen38_cuda_decode_per_bucket_wall_split_graphed_nodes_then_eager_attribution() {
    if std::env::var("NV_QWEN38_SERVING_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_QWEN38_SERVING_TEST=1 to run this probe, or NV_MODELS_ALLOW_SKIP=1 to \
                 skip it on purpose"
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_QWEN38_SERVING_TEST=1 to run");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    assert!(
        std::env::var("NV_GRAPH_OFF").ok().as_deref() != Some("1"),
        "phase A needs capture; unset NV_GRAPH_OFF (the test flips it itself for phase B)"
    );
    assert!(
        std::env::var("NV_PROF_DECODE").ok().as_deref() != Some("1"),
        "unset NV_PROF_DECODE; profiling the warm/capture passes would sync mid-capture-adjacent \
         streams and skew phase A (the test enables it only for the eager phase B)"
    );

    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, &device)
        .expect("build Qwen3.8-27B on the dense arm");
    drop(weights);
    assert!(model.is_dense(), "from_loader_dense_quantized must yield the dense arm");

    let depth = depth_from_env_default_256();
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let extra = ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
        + GRAPHED_TIMED_STEPS_32_A_MEDIAN_NOT_A_LADDER
        + ATTRIBUTED_EAGER_STEPS_12_EACH_PRINTS_A_PROF_DECODE_SPLIT
        + KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS;
    let cache_slots = depth + extra;
    let mut eng = GraphedQwen3Moe::new(model, &device, cache_slots)
        .unwrap_or_else(|e| panic!("GraphedQwen3Moe with {cache_slots} kv slots: {e:#}"));

    let chunk = SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN35_DENSE_PREFILL_BLOCK;
    let vals = fixed_seed_values_because_an_all_zero_cache_is_an_unrealistically_easy_softmax(
        chunk * n_kv * hd,
    );
    let k_template = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), &device)
        .expect("k template")
        .to_dtype(DType::BF16)
        .expect("k bf16");
    let v_template = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), &device)
        .expect("v template")
        .to_dtype(DType::BF16)
        .expect("v bf16");

    let mut pos = 0usize;
    while pos < depth {
        let n = chunk.min(depth - pos);
        if n == chunk {
            eng.prime_kv_depth_synthetically_for_ctx_timing_decode_reads_cache_size_not_values(
                &k_template,
                &v_template,
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
    eng.synchronize().expect("sync after synthetic fill");
    assert_eq!(eng.current_pos(), depth, "fill must advance current_pos like prefill");

    eng.install_grouped_moe()
        .expect("dense branch of install_grouped_moe arms capture without a dispatch");

    let mut cur = 2000u32;
    let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
        || {
            eng.forward_decode(cur)
                .unwrap_or_else(|e| panic!("graphed decode at depth {depth}: {e:#}"));
            let r = eng.logits_host().expect("logits to host");
            cur = argmax(&r);
        },
        GRAPHED_TIMED_STEPS_32_A_MEDIAN_NOT_A_LADDER,
    );
    assert!(
        eng.capture_active(),
        "phase A must run the CAPTURED decode; the engine printed its blocker just above"
    );
    let node_count = eng.captured_graph_node_count();
    assert!(
        node_count > 0,
        "capture_active with zero cached graph nodes; cached_node_count is broken"
    );
    let mut sorted = step_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let graphed_median = sorted[sorted.len() / 2];
    eprintln!(
        "DECODE-SPLIT-PHASE-A qwen38-cuda depth={depth} capture_active=true graph_nodes={node_count} graphed_median_ms_tok={graphed_median:.3} warmup_steps={warmup_steps} steps={}",
        step_ms.len()
    );

    std::env::set_var("NV_GRAPH_OFF", "1");
    std::env::set_var("NV_PROF_DECODE", "1");
    let mut eager_ms: Vec<f64> = Vec::new();
    for _ in 0..ATTRIBUTED_EAGER_STEPS_12_EACH_PRINTS_A_PROF_DECODE_SPLIT {
        let t0 = Instant::now();
        eng.forward_decode(cur)
            .unwrap_or_else(|e| panic!("eager attributed decode at depth {depth}: {e:#}"));
        let r = eng.logits_host().expect("logits to host");
        cur = argmax(&r);
        eager_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    assert!(
        !eng.capture_active() || eager_ms.is_empty(),
        "NV_GRAPH_OFF=1 must force the uncaptured path for the attributed steps"
    );
    let mut es = eager_ms.clone();
    es.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let eager_median = es[es.len() / 2];
    eprintln!(
        "DECODE-SPLIT-PHASE-B qwen38-cuda depth={depth} eager_median_ms_tok={eager_median:.3} graphed_median_ms_tok={graphed_median:.3} sync_and_launch_overhead_ms={:.3} attributed_steps={}",
        eager_median - graphed_median,
        eager_ms.len()
    );
}
