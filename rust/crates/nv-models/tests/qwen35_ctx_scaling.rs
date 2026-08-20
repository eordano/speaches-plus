#![cfg(feature = "cuda")]

mod common;
mod ctx_timing_common;
use common::argmax;
use common::ctx_tokens_from_env_default_256_8k_168k;
use common::decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state;
use common::percentile_of_sorted;
use candle_core::{DType, Device, Tensor};
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use std::time::Instant;

fn fixed_position_hashed_token_id_same_for_every_arm_so_per_position_argmax_is_comparable(
    pos: usize,
) -> u32 {
    2000 + (pos as u32 % 30000)
}

const PREFILL_CHUNK_512_QWEN3MOE_KV_IS_FLAT_WITH_NO_RING_CAP_ONLY_MAX_SEQ_BOUNDS_WRITES_512_MATCHES_THE_PROVEN_PPL_BLOCK:
    usize = 512;
const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const WARMUP_DECODE_STEPS_8_SO_THE_FIRST_ALLOCS_DONT_COUNT: usize = 8;

fn ig1_qwen35_9b_nvfp4_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN35_DENSE_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--ig1--Qwen3.5-9B-NVFP4/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("ig1 qwen3.5-9b snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.join("config.json").is_file()
                && (p.join("model.safetensors").is_file()
                    || p.join("model.safetensors.index.json").is_file())
        })
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no ig1 Qwen3.5-9B NVFP4 snapshot with weights under HOME hub; set NV_QWEN35_DENSE_DIR")
}

#[test]
#[ignore = "loads the 9.6 GiB ig1 Qwen3.5-9B NVFP4; set NV_QWEN35_CTX_TEST=1 -- decode ms/token vs KV depth (max_pos 262144 fits the full 256/8k/168k ladder), chunked-prefill primed, eager dense cuda arm (same build as NV_QWEN35_DENSE_CUDA_SERVE); run ONE depth per process via NV_CTX_TOKENS (the #96 residual race)"]
fn qwen35_9b_cuda_decode_ms_per_token_vs_context_depth_eager_dense_chunked_prefill_primed() {
    if std::env::var("NV_QWEN35_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_QWEN35_CTX_TEST != 1");
        return;
    }
    let dir = ig1_qwen35_9b_nvfp4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, &device)
        .expect("model");
    drop(weights);
    assert_eq!(
        model.dense_intermediate(),
        Some(cfg.intermediate_size),
        "from_loader_dense_quantized did not carry intermediate_size, so this is not the dense arm \
         the opt-in NV_QWEN35_DENSE_CUDA_SERVE serving path builds"
    );

    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();
    let max_pos = model.config().max_position_embeddings;
    assert!(
        max_depth + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN + 16 < max_pos,
        "requested depth {max_depth} exceeds max_position_embeddings {max_pos}"
    );

    for &depth in &depths {
        let mut cache = model
            .new_kv_cache(depth + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN + 16)
            .expect("kv cache");
        let chunk = PREFILL_CHUNK_512_QWEN3MOE_KV_IS_FLAT_WITH_NO_RING_CAP_ONLY_MAX_SEQ_BOUNDS_WRITES_512_MATCHES_THE_PROVEN_PPL_BLOCK;
        let prime_start = std::time::Instant::now();
        let mut pos = 0usize;
        while pos < depth {
            let n = chunk.min(depth - pos);
            let ids: Vec<u32> = (0..n).map(|i| 2000 + ((pos + i) as u32 % 30000)).collect();
            let tokens = Tensor::from_vec(ids, (1usize, n), &device).expect("tokens");
            let positions = Tensor::from_vec(
                (pos as i32..(pos + n) as i32).collect::<Vec<_>>(),
                n,
                &device,
            )
            .expect("positions");
            model
                .forward_with_cache(&tokens, &positions, &mut cache)
                .unwrap_or_else(|e| panic!("prefill chunk at pos {pos}: {e:#}"));
            pos += n;
        }
        let prime_s = prime_start.elapsed().as_secs_f64();

        let mut step_ms: Vec<f64> = Vec::new();
        let mut pos_argmax: Vec<(usize, u32)> = Vec::new();
        for i in 0..WARMUP_DECODE_STEPS_8_SO_THE_FIRST_ALLOCS_DONT_COUNT
            + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
        {
            let p = depth + i;
            let tokens = Tensor::from_vec(
                vec![fixed_position_hashed_token_id_same_for_every_arm_so_per_position_argmax_is_comparable(p)],
                (1usize, 1usize),
                &device,
            )
            .expect("token");
            let positions = Tensor::from_vec(vec![p as i32], 1usize, &device).expect("position");
            let t0 = std::time::Instant::now();
            let logits = model
                .forward_with_cache(&tokens, &positions, &mut cache)
                .unwrap_or_else(|e| panic!("decode at depth {p}: {e:#}"));
            let row: Vec<f32> = logits
                .to_dtype(DType::F32)
                .expect("f32")
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("sync to host");
            if i >= WARMUP_DECODE_STEPS_8_SO_THE_FIRST_ALLOCS_DONT_COUNT {
                step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            pos_argmax.push((p, argmax(&row)));
        }
        step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = step_ms[step_ms.len() / 2];
        eprintln!(
            "CTX-SCALING qwen35-9b-cuda-eager depth={depth} decode_kernel={} median_ms_tok={median:.3} tok_s={:.1} prime_s={prime_s:.1} steps={}",
            decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state(),
            1000.0 / median,
            step_ms.len()
        );
        eprintln!(
            "DECODE-ARGMAX qwen35-9b-cuda-eager depth={depth} decode_kernel={} feed=fixed_position_hashed_ids pos_argmax={pos_argmax:?}",
            decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state()
        );
    }
}

const TIMED_GRAPHED_DECODE_STEPS_64_MATCHES_THE_EAGER_LADDER: usize = 64;
const KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS: usize = 16;

const GREEDY_CHAIN_STEPS_64_LONG_ENOUGH_THAT_A_KERNEL_DEFECT_DERAILS_THE_SEQUENCE: usize = 64;

fn logits_last_row_f32(logits: &Tensor) -> Vec<f32> {
    let dims = logits.dims().to_vec();
    let seq = dims[1];
    logits
        .narrow(1, seq - 1, 1)
        .expect("last row")
        .to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("host")
}

#[test]
#[ignore = "loads the 9.6 GiB ig1 Qwen3.5-9B NVFP4; set NV_QWEN35_CTX_TEST=1 -- greedy argmax chain on a REAL chat-template prompt (synthetic-noise contexts flip argmax on near-ties and prove nothing); NV_QWEN35_CHAIN_GRAPHED=1 selects the graphed engine, NV_Q36_GRAPHED_DECODE_FIX selects the attention decode kernel; identical chains across arms are the argmax-class parity evidence"]
fn qwen35_9b_cuda_greedy_chain_real_chat_template_for_cross_arm_argmax_parity() {
    if std::env::var("NV_QWEN35_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_QWEN35_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = ig1_qwen35_9b_nvfp4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, &device)
        .expect("model");
    drop(weights);

    let tok =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let q = std::env::var("NV_QWEN35_Q")
        .unwrap_or_else(|_| "What is the capital of France? Answer in one short sentence.".into());
    let prompt_text =
        format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let prompt: Vec<u32> = tok
        .encode(prompt_text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();

    let graphed = std::env::var("NV_QWEN35_CHAIN_GRAPHED").ok().as_deref() == Some("1");
    let steps = GREEDY_CHAIN_STEPS_64_LONG_ENOUGH_THAT_A_KERNEL_DEFECT_DERAILS_THE_SEQUENCE;
    let mut chain: Vec<u32> = Vec::new();
    if graphed {
        let mut eng = GraphedQwen3Moe::new(model, &device, prompt.len() + steps + 16)
            .expect("graphed engine");
        let last = eng.prefill(&prompt).expect("prefill");
        eng.install_grouped_moe()
            .expect("dense trunk device-routing install");
        let mut cur = argmax(&last);
        for _ in 0..steps {
            chain.push(cur);
            eng.forward_decode(cur).expect("graphed decode");
            cur = argmax(&eng.logits_host().expect("logits host"));
        }
        assert!(
            eng.capture_active(),
            "graphed greedy chain fell back to uncaptured decode; the parity evidence must \
             cover the captured path"
        );
    } else {
        let mut cache = model
            .new_kv_cache(prompt.len() + steps + 16)
            .expect("kv cache");
        let k = prompt.len();
        let tokens = Tensor::from_vec(prompt.clone(), (1usize, k), &device).expect("tokens");
        let positions =
            Tensor::from_vec((0..k as i32).collect::<Vec<_>>(), k, &device).expect("positions");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .expect("prefill forward");
        let mut cur = argmax(&logits_last_row_f32(&logits));
        for step in 0..steps {
            chain.push(cur);
            let p = k + step;
            let tokens = Tensor::from_vec(vec![cur], (1usize, 1usize), &device).expect("token");
            let positions =
                Tensor::from_vec(vec![p as i32], 1usize, &device).expect("position");
            let logits = model
                .forward_with_cache(&tokens, &positions, &mut cache)
                .unwrap_or_else(|e| panic!("decode step {step}: {e:#}"));
            cur = argmax(&logits_last_row_f32(&logits));
        }
    }
    let text = tok.decode(&chain, false).unwrap_or_default();
    eprintln!(
        "GREEDY-CHAIN qwen35-9b engine={} decode_kernel={} prompt_toks={} steps={steps} ids={chain:?} text={text:?}",
        if graphed { "graphed" } else { "eager" },
        decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state(),
        prompt.len()
    );
}

#[test]
#[ignore = "loads the 9.6 GiB ig1 Qwen3.5-9B NVFP4; set NV_QWEN35_CTX_TEST=1 -- graphed serving-class decode ms/token vs KV depth for the SAME dense build, chunked-prefill primed then captured graph decode; feeds the eager ladder's fixed position-hashed ids so per-position argmax is comparable across arms; run ONE depth per process via NV_CTX_TOKENS (the #96 residual race)"]
fn qwen35_9b_cuda_serving_path_decode_ms_per_token_vs_context_depth_graphed_dense_chunk_primed() {
    if std::env::var("NV_QWEN35_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_QWEN35_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = ig1_qwen35_9b_nvfp4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, &device)
        .expect("model");
    drop(weights);
    assert_eq!(
        model.dense_intermediate(),
        Some(cfg.intermediate_size),
        "from_loader_dense_quantized did not carry intermediate_size, so this is not the dense arm \
         the opt-in NV_QWEN35_DENSE_CUDA_SERVE serving path builds"
    );

    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();
    let max_pos = model.config().max_position_embeddings;
    let per_depth_extra_slots =
        ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
            + TIMED_GRAPHED_DECODE_STEPS_64_MATCHES_THE_EAGER_LADDER
            + KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS;
    assert!(
        max_depth + per_depth_extra_slots < max_pos,
        "depth {max_depth} + decode steps exceeds max_position_embeddings {max_pos}; a model \
         declaring {max_pos} must serve this depth, so a trip here is a config bug"
    );
    let cache_slots = max_depth + per_depth_extra_slots;
    let mut eng = GraphedQwen3Moe::new(model, &device, cache_slots)
        .unwrap_or_else(|e| panic!("GraphedQwen3Moe with {cache_slots} kv slots: {e:#}"));

    for &depth in &depths {
        eng.set_moe_dispatch(None);
        eng.reset().expect("engine reset");
        assert_eq!(eng.current_pos(), 0, "reset must rewind the engine");

        let chunk = PREFILL_CHUNK_512_QWEN3MOE_KV_IS_FLAT_WITH_NO_RING_CAP_ONLY_MAX_SEQ_BOUNDS_WRITES_512_MATCHES_THE_PROVEN_PPL_BLOCK;
        let prime_start = Instant::now();
        let mut pos = 0usize;
        let mut last_row: Vec<f32> = Vec::new();
        while pos < depth {
            let n = chunk.min(depth - pos);
            let ids: Vec<u32> = (0..n)
                .map(|i| fixed_position_hashed_token_id_same_for_every_arm_so_per_position_argmax_is_comparable(pos + i))
                .collect();
            last_row = eng
                .prefill(&ids)
                .unwrap_or_else(|e| panic!("prefill chunk at pos {pos}: {e:#}"));
            pos += n;
        }
        let prime_s = prime_start.elapsed().as_secs_f64();
        assert_eq!(
            eng.current_pos(),
            depth,
            "chunked prefill must leave the engine at the primed depth"
        );
        let nan_in_last_row = last_row.iter().filter(|v| v.is_nan()).count();
        assert_eq!(
            nan_in_last_row, 0,
            "chunked dense prefill produced {nan_in_last_row} NaN logits at depth {depth}"
        );

        eng.install_grouped_moe()
            .expect("dense trunk device-routing install");

        let mut pos_argmax: Vec<(usize, u32)> = Vec::new();
        let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                let p = eng.current_pos();
                let tok = fixed_position_hashed_token_id_same_for_every_arm_so_per_position_argmax_is_comparable(p);
                eng.forward_decode(tok)
                    .unwrap_or_else(|e| panic!("graphed decode at depth {p}: {e:#}"));
                let r = eng.logits_host().expect("logits to host");
                pos_argmax.push((p, argmax(&r)));
            },
            TIMED_GRAPHED_DECODE_STEPS_64_MATCHES_THE_EAGER_LADDER,
        );
        assert!(
            eng.capture_active(),
            "the serving basis of this ladder is the CAPTURED graphed decode and the engine \
             fell back to uncaptured at depth {depth}; the engine printed its blocker to \
             stderr just above -- diagnose that blocker, do not relabel the number as \
             serving-class"
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
        eprintln!(
            "CTX-SCALING qwen35-9b-cuda-serving-graphed depth={depth} basis=graphed_dense_captured_decode_fixed_id_feed_includes_logits_host decode_kernel={} capture_active={} median_ms_tok={median:.3} p10={p10:.3} p90={p90:.3} tok_s={:.1} prime_s={prime_s:.1} prefill_chunk={chunk} steps={} warmup_steps={warmup_steps}",
            decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state(),
            eng.capture_active(),
            1000.0 / median,
            step_ms.len()
        );
        eprintln!(
            "DECODE-ARGMAX qwen35-9b-cuda-serving-graphed depth={depth} decode_kernel={} feed=fixed_position_hashed_ids pos_argmax={pos_argmax:?}",
            decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state()
        );
    }
}
