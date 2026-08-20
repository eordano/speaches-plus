#![cfg(feature = "cuda")]

mod common;
mod ctx_timing_common;
mod hub_snapshot;

use candle_core::{DType, Device, Tensor};
use common::argmax_partial_cmp as argmax;
use common::envn;
use common::prompt_for;
use nv_models::gemma4_batch_graph::BucketPlan;
use nv_models::qwen3_5_moe::qwen38_batch::Qwen38BatchLanes;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeKvCache, Qwen3_5DenseConfig};
use std::path::PathBuf;

fn solo_greedy_logits(
    model: &Qwen3Moe,
    cache: &mut Qwen3MoeKvCache,
    device: &Device,
    prompt: &[u32],
    steps: usize,
) -> Vec<Vec<f32>> {
    cache.reset();
    let (last, rest) = prompt.split_last().expect("prompt");
    let seq = rest.len();
    let tokens = Tensor::from_vec(rest.to_vec(), (1usize, seq), device).expect("tokens");
    let positions =
        Tensor::from_vec((0..seq as i32).collect::<Vec<_>>(), seq, device).expect("pos");
    model
        .forward_with_cache_dispatched_rows(&tokens, &positions, cache, None, Some(1))
        .expect("solo prefill");
    let mut out = Vec::with_capacity(steps);
    let mut t = *last;
    for _ in 0..steps {
        let pos = cache.current_len();
        let tokens = Tensor::from_vec(vec![t], (1usize, 1usize), device).expect("token");
        let positions = Tensor::from_vec(vec![pos as i32], 1usize, device).expect("pos");
        let logits = model
            .forward_with_cache_dispatched(&tokens, &positions, cache, None)
            .expect("solo decode step");
        let row: Vec<f32> = logits
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("host");
        t = argmax(&row);
        out.push(row);
    }
    out
}

#[test]
#[ignore = "loads the Qwen3.8-27B NVFP4 checkpoint; set NV_Q38_BPROF_PARITY_TEST=1 -- compares \
            the batch route (honoring NV_Q38_BATCH_GEMM) against the solo eager engine on \
            NV_Q38_BPROF_BS lanes of distinct real prompts, reporting per-step argmax \
            mismatches and the max absolute and relative logit deltas, gating nothing so the \
            drift CLASS is reported honestly"]
fn real_qwen38_27b_batch_vs_solo_argmax_and_max_logit_delta() {
    if std::env::var("NV_Q38_BPROF_PARITY_TEST").as_deref() != Ok("1") {
        panic!("set NV_Q38_BPROF_PARITY_TEST=1 to run this GPU test (it must never silently skip)");
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda with stream");
    let model = load_real_qwen38(&device);
    let vocab = model.config().vocab_size;
    let b = envn("NV_Q38_BPROF_PARITY_B", 8);
    let steps = envn("NV_Q38_RATE_STEPS", 16);
    let prefill_len = 24usize;
    let prompts: Vec<Vec<u32>> = (0..b).map(|i| prompt_for(i, prefill_len, vocab)).collect();

    let mut solo_cache = model.new_kv_cache(prefill_len + steps + 8).expect("solo cache");
    let mut refs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(b);
    for p in &prompts {
        refs.push(solo_greedy_logits(&model, &mut solo_cache, &device, p, steps));
    }
    drop(solo_cache);

    let plan = BucketPlan::new(vec![b]);
    let mut lanes = Qwen38BatchLanes::new(model, &device, prefill_len + steps + 8, plan)
        .expect("build batch lanes");
    let mut cur: Vec<Option<u32>> = Vec::with_capacity(b);
    for (i, p) in prompts.iter().enumerate() {
        lanes
            .prefill_lane(i, &p[..p.len() - 1])
            .expect("prefill lane");
        cur.push(Some(*p.last().unwrap()));
    }
    let mut mismatch_steps = 0usize;
    let mut total_steps = 0usize;
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for _ in 0..steps {
        let out = lanes.step_batch(&cur).expect("batch step");
        let step_idx = total_steps / b.max(1);
        for (lane, o) in out.iter().enumerate() {
            let got = o.as_ref().expect("active lane");
            let want = &refs[lane][step_idx];
            let ga = argmax(got);
            let wa = argmax(want);
            total_steps += 1;
            if ga != wa {
                mismatch_steps += 1;
                eprintln!(
                    "[q38-parity] lane {lane} step {step_idx}: batch argmax {ga} vs solo {wa}"
                );
            }
            for (x, y) in got.iter().zip(want.iter()) {
                let d = (x - y).abs();
                if d > max_abs {
                    max_abs = d;
                }
                let r = d / y.abs().max(1e-3);
                if r > max_rel {
                    max_rel = r;
                }
            }
            cur[lane] = Some(wa);
        }
    }
    println!(
        "Q38-PARITY b={b} steps={steps} solo_fed_tokens argmax_mismatch_steps={mismatch_steps}/{} \
         max_abs_logit_delta={max_abs:.4} max_rel_logit_delta={max_rel:.4} \
         gemm_arm={} basis=unsloth/Qwen3.8-27B-NVFP4",
        total_steps,
        std::env::var("NV_Q38_BATCH_GEMM").ok().as_deref() == Some("1"),
    );
}

fn qwen38_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
        return PathBuf::from(d);
    }
    hub_snapshot::snapshot_of("unsloth/Qwen3.8-27B-NVFP4", &["config.json", "*.safetensors"])
        .expect(
            "no hydrated unsloth/Qwen3.8-27B-NVFP4 snapshot under the HF hub roots; set NV_QWEN38_DIR",
        )
}

fn load_real_qwen38(device: &Device) -> Qwen3Moe {
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = nv_weights::QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = nv_weights::WeightLoader::open_dir(&dir, device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, device)
        .expect("build Qwen3.8-27B dense arm");
    assert!(model.is_dense(), "quantized dense loader must yield the dense arm");
    model
}

fn b_list_env_nv_q38_bprof_bs_default_1_2_4_8() -> Vec<usize> {
    std::env::var("NV_Q38_BPROF_BS")
        .unwrap_or_else(|_| "1,2,4,8".into())
        .split(',')
        .map(|t| t.trim().parse::<usize>().expect("NV_Q38_BPROF_BS"))
        .collect()
}

#[test]
#[ignore = "loads the Qwen3.8-27B NVFP4 checkpoint; set NV_Q38_BPROF_TEST=1 -- step-rate ladder \
            over NV_Q38_BPROF_BS (default 1,2,4,8) at NV_CTX_TOKENS depth; with \
            NV_Q38_BATCH_PROF=1 NV_PROF_GDN=1 NV_GRAPH_OFF=1 each step also prints per-stage \
            sync-lap shares from inside the batch step body"]
fn real_qwen38_27b_batch_step_rate_and_stage_profile() {
    if std::env::var("NV_Q38_BPROF_TEST").as_deref() != Ok("1") {
        panic!("set NV_Q38_BPROF_TEST=1 to run this GPU test (it must never silently skip)");
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda with stream");
    let model = load_real_qwen38(&device);
    let vocab = model.config().vocab_size;
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let b_list = b_list_env_nv_q38_bprof_bs_default_1_2_4_8();
    let depth = envn("NV_CTX_TOKENS", 256);
    let steps = envn("NV_Q38_RATE_STEPS", 8);
    let reps = envn("NV_Q38_RATE_REPS", 2);
    let max_seq = depth + 3 * (reps + 1) * steps + 64;
    let plan = BucketPlan::new(b_list.clone());
    let mut lanes =
        Qwen38BatchLanes::new(model, &device, max_seq, plan).expect("build batch lanes");

    let prefill_len = 23usize;
    let chunk = 512usize.min(depth.saturating_sub(prefill_len)).max(1);
    let mut state = 0x9e3779b97f4a7c15u64;
    let vals: Vec<f32> = (0..chunk * n_kv * hd)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 0.5
        })
        .collect();
    let k_t = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), &device)
        .expect("k template")
        .to_dtype(DType::BF16)
        .expect("bf16");
    let v_t = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), &device)
        .expect("v template")
        .to_dtype(DType::BF16)
        .expect("bf16");

    let prompt = prompt_for(0, prefill_len + 1, vocab);
    for lane in 0..lanes.lanes() {
        lanes
            .prefill_lane(lane, &prompt[..prompt.len() - 1])
            .expect("prefill lane");
        while lanes.lane_pos(lane) + chunk <= depth {
            lanes
                .prime_lane_kv_depth_synthetically_for_ctx_timing_decode_reads_cache_size_not_values(
                    lane, &k_t, &v_t,
                )
                .expect("prime lane");
        }
    }
    eprintln!(
        "[q38-bprof] primed {} lanes to pos {} (target depth {depth}) graph_off={} prof={}",
        lanes.lanes(),
        lanes.lane_pos(0),
        std::env::var("NV_GRAPH_OFF").ok().as_deref() == Some("1"),
        std::env::var("NV_Q38_BATCH_PROF").ok().as_deref() == Some("1"),
    );

    let seed_tok = *prompt.last().unwrap();
    let mut results: Vec<(usize, f64)> = Vec::new();
    for &bsz in &b_list {
        let mut cur: Vec<Option<u32>> = (0..bsz).map(|_| Some(seed_tok)).collect();
        let mut ms_acc: Vec<f64> = Vec::new();
        for r in 0..=reps {
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                let out = lanes.step_batch(&cur).expect("profile step");
                for (j, o) in out.iter().enumerate() {
                    cur[j] = Some(argmax(o.as_ref().unwrap()));
                }
            }
            lanes.synchronize().expect("sync");
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
            if r == 0 {
                eprintln!("[q38-bprof] B={bsz} warmup rep discarded: {ms:.2} ms/step");
                continue;
            }
            ms_acc.push(ms);
        }
        let mean = ms_acc.iter().sum::<f64>() / ms_acc.len() as f64;
        results.push((bsz, mean));
        eprintln!(
            "[q38-bprof] B={bsz} depth={} step {mean:.2} ms = {:.2} ms/lane-tok | aggregate {:.1} tok/s (captures={} replays={})",
            lanes.lane_pos(0),
            mean / bsz as f64,
            bsz as f64 * 1000.0 / mean,
            lanes.captures(),
            lanes.replays()
        );
    }
    let b1 = results.iter().find(|(b, _)| *b == 1).map(|(_, m)| *m);
    for (bsz, ms) in &results {
        let vs = b1
            .map(|b| format!("{:.2}x aggregate vs B=1", b * *bsz as f64 / ms))
            .unwrap_or_else(|| "no B=1 in ladder".into());
        eprintln!(
            "[q38-bprof] SUMMARY B={bsz}: {ms:.2} ms/step, {vs}, basis: unsloth/Qwen3.8-27B-NVFP4, synthetic prime depth {}, {steps} timed steps x {reps} reps",
            lanes.lane_pos(0),
        );
    }
}

const A_FLAT_MS_CURVE_ACROSS_PROMPT_LENGTH_MEANS_THE_PREFILL_WALL_IS_PER_CALL_NOT_PER_TOKEN_AND_THE_FIX_IS_ONE_WIDER_CALL_NOT_EIGHT_CONCURRENT_ONES:
    &str = "group formation pays one prefill_lane per lane. If a 20-token prefill and a \
            160-token prefill cost the same wall, the 27B prefill at chat-prompt length is \
            weight-traffic bound and every lane's rows are free capacity in one call, so the \
            wall collapses by widening m, not by forking eight streams. If instead the curve \
            is linear in tokens, the call is already compute bound and only real concurrency \
            or a faster kernel moves it";

fn prompt_lengths_env_nv_q38_prefill_lens_default_20_40_80_160_320() -> Vec<usize> {
    std::env::var("NV_Q38_PREFILL_LENS")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![20, 40, 80, 160, 320])
}

#[test]
#[ignore = "loads the Qwen3.8-27B NVFP4 checkpoint; set NV_Q38_PREFILL_PROBE=1 -- times one \
            prefill_lane per prompt length in NV_Q38_PREFILL_LENS (default 20,40,80,160,320) \
            and prints ms and ms/token, so the group-formation wall is attributed to per-call \
            weight traffic or to per-token compute before anything is built to hide it"]
fn real_qwen38_27b_prefill_wall_is_per_call_or_per_token() {
    if std::env::var("NV_Q38_PREFILL_PROBE").as_deref() != Ok("1") {
        panic!("set NV_Q38_PREFILL_PROBE=1 to run this GPU test (it must never silently skip)");
    }
    eprintln!("[q38-prefill-probe] {A_FLAT_MS_CURVE_ACROSS_PROMPT_LENGTH_MEANS_THE_PREFILL_WALL_IS_PER_CALL_NOT_PER_TOKEN_AND_THE_FIX_IS_ONE_WIDER_CALL_NOT_EIGHT_CONCURRENT_ONES}");
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda with stream");
    let model = load_real_qwen38(&device);
    let vocab = model.config().vocab_size;
    let lens = prompt_lengths_env_nv_q38_prefill_lens_default_20_40_80_160_320();
    let max_len = lens.iter().copied().max().unwrap_or(320);
    let plan = BucketPlan::new(vec![1usize]);
    let mut lanes = Qwen38BatchLanes::new(model, &device, max_len + 64, plan)
        .expect("build batch lanes");
    let warm = prompt_for(7, 24, vocab);
    lanes.prefill_lane(0, &warm).expect("warm prefill");
    for len in lens {
        let prompt = prompt_for(11, len, vocab);
        let mut best = f64::INFINITY;
        for _ in 0..2 {
            let t0 = std::time::Instant::now();
            lanes.prefill_lane(0, &prompt).expect("probe prefill");
            lanes.synchronize().expect("sync");
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
        }
        eprintln!(
            "[q38-prefill-probe] seq={len} {best:.1} ms = {:.2} ms/token",
            best / len as f64
        );
    }
}
