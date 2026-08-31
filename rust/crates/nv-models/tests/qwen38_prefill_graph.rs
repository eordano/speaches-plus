#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::time::Instant;

mod common;
mod ctx_timing_common;
mod hub_snapshot;
use common::qwen38_snapshot_dir_env_override_then_home_hub;

const KV_HEADROOM_8_ROWS_KEEPS_THE_CACHE_STRICTLY_LARGER_THAN_THE_PROMPT: usize = 8;
const BITWISE_DEPTHS_FULL_8192_AND_PARTIAL_10240_COVER_REPLAY_AND_THE_EAGER_LAST_CHUNK: &[usize] =
    &[8192, 10240];
const EXPECTED_SEGMENTS_16_THE_QWEN38_LAYER_MAP_IS_16_RUNS_OF_3_GDN_LAYERS_BETWEEN_FULL_ATTENTION:
    usize = 16;

fn depths_from_env(default: &[usize]) -> Vec<usize> {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => v
            .split(',')
            .map(|s| {
                let s = s.trim();
                let (num, mult) = match s.strip_suffix(['k', 'K']) {
                    Some(n) => (n, 1024usize),
                    None => (s, 1usize),
                };
                num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
            })
            .collect(),
        Err(_) => default.to_vec(),
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn gate_env_or_allow_skip() -> bool {
    if std::env::var("NV_QWEN38_SERVING_TEST").ok().as_deref() == Some("1") {
        return true;
    }
    if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
        panic!(
            "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_QWEN38_SERVING_TEST=1, or \
             NV_MODELS_ALLOW_SKIP=1 to skip on purpose"
        );
    }
    eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_QWEN38_SERVING_TEST=1 to run");
    false
}

fn build_model(device: &Device) -> (Qwen3Moe, String) {
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let rev = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, device)
        .expect("build Qwen3.8-27B on the eager cuda dense arm");
    assert!(
        model.is_dense(),
        "from_loader_dense_quantized must yield the dense arm"
    );
    (model, rev)
}

fn prompt_ids(depth: usize) -> Vec<u32> {
    (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect()
}

fn last_row_bits(logits: &Tensor) -> Vec<u32> {
    let (_b, s, _v) = logits.dims3().expect("logits dims");
    logits
        .narrow(1, s - 1, 1)
        .expect("narrow last row")
        .to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to host")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 dense arm; set NV_QWEN38_SERVING_TEST=1 -- the \
            NV_Q38_PREFILL_GRAPH bitwise gate: one model, eager chunked prefill vs the graphed \
            session run twice on fresh caches (run 1 warms one eager chunk then captures, run 2 \
            is pure segment replay), last-row logits compared bit-for-bit at a full-chunk depth \
            and a partial-last-chunk depth (NV_CTX_TOKENS, default 8192,10240); also pins the \
            captured segment count to the 16 GDN runs of the qwen3.8 layer map"]
fn qwen38_cuda_prefill_graph_last_row_logits_bitwise_vs_eager_chunks() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let (model, _rev) = build_model(&device);
    let chunk = nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
        .expect("this gate requires chunked prefill; unset NV_Q38_PREFILL_CHUNK=0");
    let mut sess = model
        .new_prefill_graph_session(chunk)
        .expect("prefill graph session");
    for depth in depths_from_env(BITWISE_DEPTHS_FULL_8192_AND_PARTIAL_10240_COVER_REPLAY_AND_THE_EAGER_LAST_CHUNK)
    {
        assert!(
            depth > chunk,
            "depth {depth} must exceed chunk {chunk} for a chunked-loop gate"
        );
        let ids = prompt_ids(depth);
        let max_seq = depth + KV_HEADROOM_8_ROWS_KEEPS_THE_CACHE_STRICTLY_LARGER_THAN_THE_PROMPT;

        let mut cache = model.new_kv_cache(max_seq).expect("kv cache eager");
        let eager = model
            .prefill_chunks_dispatched_serving_last_row_logits_bounding_transients_to_one_chunk(
                &ids, chunk, &mut cache, None,
            )
            .expect("eager chunked prefill");
        device.synchronize().expect("sync eager");
        let eager_bits = last_row_bits(&eager);
        drop(cache);

        for run in 0..2usize {
            let mut cache = model.new_kv_cache(max_seq).expect("kv cache graphed");
            let graphed = model
                .prefill_chunks_graphed_serving_last_row_logits_replaying_gdn_mlp_segments_with_full_attention_eager_breaks(
                    &ids, chunk, &mut cache, &mut sess,
                )
                .unwrap_or_else(|e| panic!("graphed prefill run {run} depth {depth}: {e:#}"));
            device.synchronize().expect("sync graphed");
            let bits = last_row_bits(&graphed);
            assert_eq!(cache.current_len(), depth, "graphed prefill must commit all tokens");
            let diff = eager_bits
                .iter()
                .zip(bits.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                diff,
                0,
                "graphed prefill run {run} moved {diff}/{} last-row logit bits at depth {depth} \
                 chunk {chunk}",
                bits.len()
            );
        }
        let segs = sess.captured_segment_count();
        assert_eq!(
            segs,
            EXPECTED_SEGMENTS_16_THE_QWEN38_LAYER_MAP_IS_16_RUNS_OF_3_GDN_LAYERS_BETWEEN_FULL_ATTENTION,
            "captured segment count {segs} does not match the qwen3.8 layer map"
        );
        println!(
            "QWEN38-PFGRAPH-BITWISE depth={depth} chunk={chunk} segments={segs} \
             logit_bits_identical={}",
            eager_bits.len()
        );
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 dense arm; set NV_QWEN38_SERVING_TEST=1 -- the \
            host-enqueue-share instrument for the graphed prefill, same clock methodology as \
            qwen38_prefill_seam.rs: host_enqueue_ms = elapsed when the prefill call returns \
            (everything enqueued), wall_ms = elapsed after the device sync; one warm pass per arm \
            then one measured pass, eager arm then graphed arm in the same process; depths via \
            NV_CTX_TOKENS (default 8192,32768)"]
fn qwen38_cuda_prefill_graph_host_enqueue_share_vs_eager() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let (model, _rev) = build_model(&device);
    let chunk = nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
        .expect("this instrument requires chunked prefill; unset NV_Q38_PREFILL_CHUNK=0");
    let mut sess = model
        .new_prefill_graph_session(chunk)
        .expect("prefill graph session");
    for depth in depths_from_env(&[8192, 32768]) {
        let ids = prompt_ids(depth);
        let max_seq = depth + KV_HEADROOM_8_ROWS_KEEPS_THE_CACHE_STRICTLY_LARGER_THAN_THE_PROMPT;
        let mut measure = |label: &str, graphed: bool| {
            let mut row = (0f64, 0f64);
            for run in 0..2usize {
                let mut cache = model.new_kv_cache(max_seq).expect("kv cache");
                device.synchronize().expect("sync before clock");
                let t0 = Instant::now();
                let logits = if graphed {
                    model
                        .prefill_chunks_graphed_serving_last_row_logits_replaying_gdn_mlp_segments_with_full_attention_eager_breaks(
                            &ids, chunk, &mut cache, &mut sess,
                        )
                        .expect("graphed prefill")
                } else {
                    model
                        .prefill_chunks_dispatched_serving_last_row_logits_bounding_transients_to_one_chunk(
                            &ids, chunk, &mut cache, None,
                        )
                        .expect("eager prefill")
                };
                let host_enqueue_ms = t0.elapsed().as_secs_f64() * 1e3;
                device.synchronize().expect("sync after enqueue");
                let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
                let finite = last_row_bits(&logits)
                    .iter()
                    .filter(|b| f32::from_bits(**b).is_finite())
                    .count();
                assert!(finite > 0, "{label} produced no finite logits at depth {depth}");
                if run == 1 {
                    row = (host_enqueue_ms, wall_ms);
                }
            }
            row
        };
        let (eager_host, eager_wall) = measure("eager", false);
        let (graph_host, graph_wall) = measure("graphed", true);
        println!(
            "QWEN38-PFGRAPH-HOSTSHARE depth={depth} chunk={chunk} \
             eager_host_enqueue_ms={eager_host:.1} eager_wall_ms={eager_wall:.1} \
             eager_host_share_pct={:.2} graph_host_enqueue_ms={graph_host:.1} \
             graph_wall_ms={graph_wall:.1} graph_host_share_pct={:.2} \
             segments={}",
            100.0 * eager_host / eager_wall,
            100.0 * graph_host / graph_wall,
            sess.captured_segment_count()
        );
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 dense arm; set NV_QWEN38_SERVING_TEST=1 -- \
            serving-semantic GRAPHED chunked prefill wall vs prompt depth, the pfgraph ON arm of \
            the depth A/B (the OFF arm is qwen38_prefill_depth.rs's \
            qwen38_cuda_chunked_prefill_tok_s_vs_prompt_depth_serving_last_row in its own \
            process): one session, run 0 warms and captures, rows print for warm runs only; \
            depths via NV_CTX_TOKENS (default 8192,32768), runs via NV_PREFILL_WARMUP_RUNS/\
            NV_PREFILL_TIMED_RUNS"]
fn qwen38_cuda_prefill_graph_tok_s_vs_prompt_depth_serving_last_row() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let (model, rev) = build_model(&device);
    let n_layers = model.config().num_hidden_layers;
    let chunk = nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
        .expect("this harness requires chunked prefill; unset NV_Q38_PREFILL_CHUNK=0");
    let mut sess = model
        .new_prefill_graph_session(chunk)
        .expect("prefill graph session");
    let warmup = env_usize("NV_PREFILL_WARMUP_RUNS", 1);
    let timed = env_usize("NV_PREFILL_TIMED_RUNS", 2);
    assert!(timed >= 1, "NV_PREFILL_TIMED_RUNS must be >= 1");
    for depth in depths_from_env(&[8192, 32768]) {
        let ids = prompt_ids(depth);
        let max_seq = depth + KV_HEADROOM_8_ROWS_KEEPS_THE_CACHE_STRICTLY_LARGER_THAN_THE_PROMPT;
        for run in 0..warmup + timed {
            let mut cache = model.new_kv_cache(max_seq).expect("kv cache");
            device.synchronize().expect("sync before starting the clock");
            let t0 = Instant::now();
            let logits = model
                .prefill_chunks_graphed_serving_last_row_logits_replaying_gdn_mlp_segments_with_full_attention_eager_breaks(
                    &ids, chunk, &mut cache, &mut sess,
                )
                .unwrap_or_else(|e| panic!("graphed prefill depth {depth} run {run}: {e:#}"));
            device.synchronize().expect("sync before stopping the clock");
            let dt = t0.elapsed().as_secs_f64();
            assert_eq!(cache.current_len(), depth, "prefill must commit all {depth} tokens");
            let bits = last_row_bits(&logits);
            let finite = bits.iter().filter(|b| f32::from_bits(**b).is_finite()).count();
            assert_eq!(finite, bits.len(), "non-finite logits at depth {depth}");
            let tok_s = depth as f64 / dt;
            if run >= warmup {
                println!(
                    "QWEN38-CUDA-PREFILL checkpoint=unsloth/Qwen3.8-27B-NVFP4@{rev} backend=cuda \
                     layers={n_layers} prompt_tokens={depth} prefill_s={dt:.2} tok_s={tok_s:.1} \
                     chunk={chunk} arm=pfgraph segments={} run={run}",
                    sess.captured_segment_count()
                );
            } else {
                eprintln!(
                    "qwen38-cuda-pfgraph warmup depth={depth} run={run} prefill_s={dt:.2} \
                     tok_s={tok_s:.1} chunk={chunk}"
                );
            }
        }
    }
}
