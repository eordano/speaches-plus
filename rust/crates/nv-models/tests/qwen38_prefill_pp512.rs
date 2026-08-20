#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use std::time::Instant;

mod ctx_timing_common;
mod hub_snapshot;
mod common;
use common::qwen38_snapshot_dir_env_override_then_home_hub;

const PP_CHUNK_512_MATCHES_LLAMA_BENCH_PP512_AND_THE_PROVEN_PREFILL_BLOCK: usize = 512;
const TIMED_RUNS_2_THE_HONEST_MINIMUM_FOR_A_STDDEV_FREE_MACHINE_LINE: usize = 2;
const WARMUP_RUNS_1_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW: usize = 1;

fn gate_env_or_allow_skip() -> bool {
    if std::env::var("NV_QWEN38_SERVING_TEST").ok().as_deref() == Some("1") {
        return true;
    }
    if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
        panic!(
            "set NV_QWEN38_SERVING_TEST=1 to run this gate, or NV_MODELS_ALLOW_SKIP=1 to \
             skip it on purpose; a silently-ok prefill gate would hide exactly the failure \
             class task #106 exists to catch"
        );
    }
    eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_QWEN38_SERVING_TEST=1 to run");
    false
}

fn load_dense_arm(device: &Device) -> Qwen3Moe {
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
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
    model
}

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 dense arm; set NV_QWEN38_SERVING_TEST=1 -- the pp512 prefill gate: one 512-token eager prefill from an empty cache per run, fresh cache each run the way a server prompt starts, 1 untimed warmup run then 2 timed runs; the basis includes lm_head over ALL 512 positions (this engine's prefill computes it; llama-bench's pp512 does not), so the printed tok/s is the honest lower bound vs the 2569 record bar"]
fn qwen38_cuda_dense_prefill_pp512_tok_s_eager_fresh_cache() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let model = load_dense_arm(&device);

    let chunk = PP_CHUNK_512_MATCHES_LLAMA_BENCH_PP512_AND_THE_PROVEN_PREFILL_BLOCK;
    let ids: Vec<u32> = (0..chunk).map(|i| 2000 + (i as u32 % 30000)).collect();
    let positions_v: Vec<i32> = (0..chunk as i32).collect();

    let mut timed_tok_s: Vec<f64> = Vec::new();
    let total_runs = WARMUP_RUNS_1_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW
        + TIMED_RUNS_2_THE_HONEST_MINIMUM_FOR_A_STDDEV_FREE_MACHINE_LINE;
    for run in 0..total_runs {
        let mut cache = model.new_kv_cache(chunk).expect("kv cache");
        let tokens =
            Tensor::from_vec(ids.clone(), (1usize, chunk), &device).expect("tokens");
        let positions =
            Tensor::from_vec(positions_v.clone(), chunk, &device).expect("positions");
        device.synchronize().expect("sync before starting the clock");
        let t0 = Instant::now();
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .unwrap_or_else(|e| panic!("pp512 prefill run {run}: {e:#}"));
        device.synchronize().expect("sync before stopping the clock");
        let dt = t0.elapsed().as_secs_f64();
        assert_eq!(cache.current_len(), chunk, "prefill must commit all {chunk} tokens");
        let last_row: Vec<f32> = logits
            .narrow(1, chunk - 1, 1)
            .expect("last row")
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to host");
        let finite = last_row.iter().filter(|v| v.is_finite()).count();
        assert_eq!(
            finite,
            last_row.len(),
            "pp512 produced non-finite logits; a NaN prefill would make the tok/s meaningless"
        );
        let tok_s = chunk as f64 / dt;
        let warm = run >= WARMUP_RUNS_1_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW;
        if warm {
            timed_tok_s.push(tok_s);
        }
        eprintln!(
            "PP qwen38-cuda pp512 run={run} timed={warm} basis=eager_dense_chunked_prefill_forward_with_cache_lm_head_all_positions_fresh_cache tok_s={tok_s:.1} ms={:.1}",
            dt * 1e3
        );
    }
    let mean = timed_tok_s.iter().sum::<f64>() / timed_tok_s.len() as f64;
    assert!(
        mean.is_finite() && mean > 0.0,
        "mean pp512 tok/s is not a positive number: {mean}"
    );
    eprintln!(
        "PP qwen38-cuda pp512 basis=eager_dense_chunked_prefill_forward_with_cache_lm_head_all_positions_fresh_cache runs={} mean_tok_s={mean:.1} run_tok_s={:?}",
        timed_tok_s.len(),
        timed_tok_s
            .iter()
            .map(|v| (v * 10.0).round() / 10.0)
            .collect::<Vec<f64>>()
    );
}

const CHUNK_PARITY_CONTROL_MARGIN_3X_OVER_THE_SAME_CODE_RECHUNKING_NOISE_FLOOR: f64 = 3.0;
const CHUNK_PARITY_AVG_NLL_DELTA_FLOOR_0_01_NATS_A_STRIDE_DEFECT_MOVES_WHOLE_NATS: f64 = 0.01;
const CHUNK_PARITY_TOP1_FLIP_FLOOR_8_OF_512_THE_BF16_RESIDUAL_CASCADE_BAND: usize = 8;

const SCAN_PARITY_T_512_MATCHES_THE_PP512_CHUNK: usize = 512;
const SCAN_PARITY_REL_TOL_5E3_BOUNDS_FMA_VS_SPLIT_MUL_ADD_DRIFT_THROUGH_THE_512_STEP_RECURRENCE:
    f32 = 5e-3;

#[test]
fn gdn_chunk_scan_tracks_the_candle_scan_on_a_synthetic_512_token_layer() {
    use nv_layers::linear::Linear;
    use nv_layers::linear_attn::{LinAttnState, LinearAttention, LinearAttentionConfig};

    let Ok(device) = Device::new_cuda_with_stream(0) else {
        eprintln!("[skip] no cuda device; the chunk scan parity needs the card");
        return;
    };
    let cfg = LinearAttentionConfig {
        hidden_size: 256,
        linear_num_key_heads: 16,
        linear_num_value_heads: 48,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        mamba_ssm_dtype: DType::F32,
        rms_eps: 1e-6,
    };
    let hidden = cfg.hidden_size;
    let conv_dim = cfg.conv_dim();
    let value_dim = cfg.value_dim();
    let n_v = cfg.linear_num_value_heads;
    let bf = |t: Tensor| t.to_dtype(DType::BF16).expect("bf16 cast");
    let rnd = |shape: &[usize]| bf(Tensor::randn(0f32, 0.25f32, shape, &device).expect("randn"));
    let la = LinearAttention::new(
        cfg,
        Linear::new(rnd(&[conv_dim, hidden]), None).expect("qkv"),
        Linear::new(rnd(&[value_dim, hidden]), None).expect("z"),
        Linear::new(rnd(&[n_v, hidden]), None).expect("a"),
        Linear::new(rnd(&[n_v, hidden]), None).expect("b"),
        rnd(&[conv_dim, 1, cfg.linear_conv_kernel_dim]),
        rnd(&[n_v]),
        rnd(&[n_v]),
        rnd(&[cfg.linear_value_head_dim]),
        Linear::new(rnd(&[hidden, value_dim]), None).expect("out"),
    )
    .expect("synthetic GDN layer with the q38 head count and head dims");

    let t = SCAN_PARITY_T_512_MATCHES_THE_PP512_CHUNK;
    let x = bf(Tensor::randn(0f32, 1.0f32, &[1usize, t, hidden][..], &device).expect("randn x"));
    let host = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("host")
    };
    let run_arm = |on: bool| -> (Vec<f32>, Vec<f32>) {
        if on {
            std::env::set_var("NV_Q38_GDN_CHUNK_PREFILL", "1");
        } else {
            std::env::remove_var("NV_Q38_GDN_CHUNK_PREFILL");
        }
        let mut state: Option<LinAttnState> = None;
        let out = la
            .forward_with_state(&x, &mut state)
            .expect("stateful forward");
        device.synchronize().expect("sync");
        let st = state.expect("scan must leave a state");
        (host(&out), host(st.recurrent_state()))
    };
    let (out_eager, st_eager) = run_arm(false);
    let (out_chunk, st_chunk) = run_arm(true);
    std::env::remove_var("NV_Q38_GDN_CHUNK_PREFILL");

    let max_abs = |a: &[f32], b: &[f32]| -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    let max_mag = |a: &[f32]| a.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d_out = max_abs(&out_eager, &out_chunk);
    let d_st = max_abs(&st_eager, &st_chunk);
    let m_out = max_mag(&out_eager).max(1e-6);
    let m_st = max_mag(&st_eager).max(1e-6);
    eprintln!(
        "PP qwen38 gdn-chunk-scan synthetic parity basis=(q38 head geometry, t={t}, fresh state, \
         arm A=candle token-sequential scan, arm B=fused conv+qknorm+scan+gate CUDA path): \
         out max_abs={d_out:.6} max_mag={m_out:.4} rel={:.6}; final_state max_abs={d_st:.6} \
         max_mag={m_st:.4} rel={:.6}",
        d_out / m_out,
        d_st / m_st
    );
    let tol =
        SCAN_PARITY_REL_TOL_5E3_BOUNDS_FMA_VS_SPLIT_MUL_ADD_DRIFT_THROUGH_THE_512_STEP_RECURRENCE;
    assert!(
        d_out / m_out <= tol && d_st / m_st <= tol,
        "chunk scan diverged from the candle scan beyond fp-reorder drift: out rel {} state rel \
         {} tol {tol}",
        d_out / m_out,
        d_st / m_st
    );
}

fn natural_text_512_ids_because_random_ids_make_every_next_token_a_near_tie(
    dir: &std::path::Path,
    n: usize,
) -> Vec<u32> {
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .expect("tokenizer.json ships with the checkpoint");
    let para = "The lighthouse keeper climbed the spiral staircase every evening at dusk, \
                carrying a small brass lamp and a logbook whose pages had softened with salt \
                air. From the gallery he could see the fishing boats returning to the harbor, \
                their lanterns swaying as the tide pushed against the breakwater. He recorded \
                the wind, the visibility, and the number of ships that passed the point, and \
                when the fog rolled in he wound the great clockwork horn and listened to its \
                voice roll out across the water. ";
    let mut text = String::new();
    while text.len() < n * 8 {
        text.push_str(para);
    }
    let ids = tok
        .encode(text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert!(
        ids.len() >= n,
        "natural text yielded {} tokens, need {n}",
        ids.len()
    );
    ids[..n].to_vec()
}

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 dense arm; set NV_QWEN38_SERVING_TEST=1 -- the chunked GDN prefill parity gate: teacher-forced 512 natural-text rows through forward_with_cache with lm_head over all positions, three arms: eager candle scan (reference), NV_Q38_GDN_CHUNK_PREFILL=1 (treatment), and the same eager code re-chunked 2x256 (control measuring the codebase's own accepted fp-reorder noise floor -- through 64 bf16 residual layers any batching change flips near-tie argmaxes, the same phenomenon that moved the MTP v3 bar to draft-policy invariance); the treatment must stay within 3x the control on teacher-forced avg NLL delta and top-1 flip count (floored at 0.01 nats / 8 rows), because a real defect -- wrong row, stale state, stride bug -- moves NLL by whole nats, not centinats"]
fn qwen38_cuda_gdn_chunk_prefill_parity_teacher_forced_512_rows_vs_eager_scan() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let model = load_dense_arm(&device);

    let chunk = PP_CHUNK_512_MATCHES_LLAMA_BENCH_PP512_AND_THE_PROVEN_PREFILL_BLOCK;
    let ids = natural_text_512_ids_because_random_ids_make_every_next_token_a_near_tie(&dir, chunk);
    let positions_v: Vec<i32> = (0..chunk as i32).collect();

    let run_arm = |label: &str, chunked_env_on: bool| -> Vec<f32> {
        if chunked_env_on {
            std::env::set_var("NV_Q38_GDN_CHUNK_PREFILL", "1");
        } else {
            std::env::remove_var("NV_Q38_GDN_CHUNK_PREFILL");
        }
        let mut cache = model.new_kv_cache(chunk).expect("kv cache");
        let tokens = Tensor::from_vec(ids.clone(), (1usize, chunk), &device).expect("tokens");
        let positions =
            Tensor::from_vec(positions_v.clone(), chunk, &device).expect("positions");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .unwrap_or_else(|e| panic!("parity arm {label}: {e:#}"));
        device.synchronize().expect("sync after parity arm");
        assert_eq!(cache.current_len(), chunk, "arm {label} must commit all {chunk} tokens");
        logits
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to host")
    };

    let run_split_control = || -> Vec<f32> {
        std::env::remove_var("NV_Q38_GDN_CHUNK_PREFILL");
        let mut cache = model.new_kv_cache(chunk).expect("kv cache");
        let mut all: Vec<f32> = Vec::new();
        let half = chunk / 2;
        for start in [0usize, half] {
            let tokens =
                Tensor::from_vec(ids[start..start + half].to_vec(), (1usize, half), &device)
                    .expect("tokens");
            let positions = Tensor::from_vec(
                (start as i32..(start + half) as i32).collect::<Vec<i32>>(),
                half,
                &device,
            )
            .expect("positions");
            let logits = model
                .forward_with_cache(&tokens, &positions, &mut cache)
                .unwrap_or_else(|e| panic!("split control arm: {e:#}"));
            device.synchronize().expect("sync split arm");
            all.extend(
                logits
                    .to_dtype(DType::F32)
                    .expect("f32")
                    .flatten_all()
                    .expect("flatten")
                    .to_vec1::<f32>()
                    .expect("to host"),
            );
        }
        all
    };

    let eager = run_arm("eager_candle_scan", false);
    let chunked = run_arm("cuda_chunk_scan", true);
    let split_control = run_split_control();
    std::env::remove_var("NV_Q38_GDN_CHUNK_PREFILL");
    assert_eq!(eager.len(), chunked.len(), "arms returned different logit counts");
    assert_eq!(eager.len(), split_control.len(), "split control logit count mismatch");
    let vocab = eager.len() / chunk;

    let avg_nll = |logits: &[f32]| -> f64 {
        let mut acc = 0f64;
        for row in 0..chunk - 1 {
            let r = &logits[row * vocab..(row + 1) * vocab];
            let m = r.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v)) as f64;
            let lse = m + r.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln();
            acc += lse - r[ids[row + 1] as usize] as f64;
        }
        acc / (chunk - 1) as f64
    };
    let compare = |a: &[f32], b: &[f32]| -> (f32, Vec<(usize, f32)>) {
        let mut max_abs = 0f32;
        let mut flips: Vec<(usize, f32)> = Vec::new();
        for row in 0..chunk {
            let ra = &a[row * vocab..(row + 1) * vocab];
            let rb = &b[row * vocab..(row + 1) * vocab];
            let mut a_top = 0usize;
            let mut b_top = 0usize;
            let mut a_best = f32::NEG_INFINITY;
            let mut a_second = f32::NEG_INFINITY;
            let mut b_best = f32::NEG_INFINITY;
            for i in 0..vocab {
                assert!(
                    ra[i].is_finite() && rb[i].is_finite(),
                    "non-finite logit at row {row} col {i}"
                );
                let d = (ra[i] - rb[i]).abs();
                if d > max_abs {
                    max_abs = d;
                }
                if ra[i] > a_best {
                    a_second = a_best;
                    a_best = ra[i];
                    a_top = i;
                } else if ra[i] > a_second {
                    a_second = ra[i];
                }
                if rb[i] > b_best {
                    b_best = rb[i];
                    b_top = i;
                }
            }
            if a_top != b_top {
                flips.push((row, a_best - a_second));
            }
        }
        (max_abs, flips)
    };

    let nll_eager = avg_nll(&eager);
    let nll_chunked = avg_nll(&chunked);
    let nll_split = avg_nll(&split_control);
    let (max_abs_chunk, flips_chunk) = compare(&eager, &chunked);
    let (max_abs_split, flips_split) = compare(&eager, &split_control);
    let nll_delta_chunk = (nll_eager - nll_chunked).abs();
    let nll_delta_split = (nll_eager - nll_split).abs();
    eprintln!(
        "PP qwen38-cuda gdn-chunk-prefill parity basis=teacher_forced_512_rows_natural_text_forward_with_cache_all_positions_fresh_cache rows={chunk} vocab={vocab} avg_nll: eager={nll_eager:.5} chunked={nll_chunked:.5} split_control={nll_split:.5} | chunked_vs_eager: nll_delta={nll_delta_chunk:.5} top1_flips={} max_abs={max_abs_chunk:.4} | split_control_vs_eager (same code, 2x256 chunking): nll_delta={nll_delta_split:.5} top1_flips={} max_abs={max_abs_split:.4} | chunk_flip_rows={flips_chunk:?} split_flip_rows={flips_split:?}",
        flips_chunk.len(),
        flips_split.len()
    );
    let nll_bar = (CHUNK_PARITY_CONTROL_MARGIN_3X_OVER_THE_SAME_CODE_RECHUNKING_NOISE_FLOOR
        * nll_delta_split)
        .max(CHUNK_PARITY_AVG_NLL_DELTA_FLOOR_0_01_NATS_A_STRIDE_DEFECT_MOVES_WHOLE_NATS);
    assert!(
        nll_delta_chunk <= nll_bar,
        "teacher-forced avg NLL moved {nll_delta_chunk:.5} nats vs eager, above the bar \
         {nll_bar:.5} (3x the same-code 2x256 rechunking control at {nll_delta_split:.5}, floored \
         at 0.01); a wrong-row/stale-state/stride defect moves whole nats"
    );
    let flip_bar = (CHUNK_PARITY_CONTROL_MARGIN_3X_OVER_THE_SAME_CODE_RECHUNKING_NOISE_FLOOR
        as usize
        * flips_split.len())
        .max(CHUNK_PARITY_TOP1_FLIP_FLOOR_8_OF_512_THE_BF16_RESIDUAL_CASCADE_BAND);
    assert!(
        flips_chunk.len() <= flip_bar,
        "top-1 flipped on {} of {chunk} rows vs eager, above the bar {flip_bar} (3x the \
         same-code rechunking control at {}, floored at 8); rows {flips_chunk:?}",
        flips_chunk.len(),
        flips_split.len()
    );
}

const SERVING_CHUNK_SWEEP_128_256_512_LOCATES_THE_ATTENTION_VS_GEMM_CROSSOVER: [usize; 3] =
    [512, 256, 128];

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 dense arm; set NV_QWEN38_SERVING_TEST=1 -- the serving-semantic pp512 gate: chat prefill samples only the final position, so lm_head runs on the last row per chunk (teacher-forced scorers keep the all-positions path via forward_with_cache); sweeps chunk sizes 512/256/128 over the same 512-token prompt, fresh cache per run, 1 untimed warmup then 2 timed runs per chunk size; NV_PROF_PREFILL=1 adds synced wall splits per bucket"]
fn qwen38_cuda_dense_prefill_pp512_tok_s_serving_last_row_chunk_sweep() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let model = load_dense_arm(&device);

    let total = PP_CHUNK_512_MATCHES_LLAMA_BENCH_PP512_AND_THE_PROVEN_PREFILL_BLOCK;
    let ids: Vec<u32> = (0..total).map(|i| 2000 + (i as u32 % 30000)).collect();
    let total_runs = WARMUP_RUNS_1_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW
        + TIMED_RUNS_2_THE_HONEST_MINIMUM_FOR_A_STDDEV_FREE_MACHINE_LINE;

    for &chunk in &SERVING_CHUNK_SWEEP_128_256_512_LOCATES_THE_ATTENTION_VS_GEMM_CROSSOVER {
        assert_eq!(total % chunk, 0, "chunk {chunk} must tile the {total}-token prompt");
        let mut timed_tok_s: Vec<f64> = Vec::new();
        for run in 0..total_runs {
            let mut cache = model.new_kv_cache(total).expect("kv cache");
            device.synchronize().expect("sync before starting the clock");
            let t0 = Instant::now();
            let mut last_logits = None;
            for start in (0..total).step_by(chunk) {
                let tokens =
                    Tensor::from_vec(ids[start..start + chunk].to_vec(), (1usize, chunk), &device)
                        .expect("tokens");
                let positions_v: Vec<i32> = (start as i32..(start + chunk) as i32).collect();
                let positions =
                    Tensor::from_vec(positions_v, chunk, &device).expect("positions");
                let logits = model
                    .forward_with_cache_dispatched_rows(
                        &tokens,
                        &positions,
                        &mut cache,
                        None,
                        Some(1),
                    )
                    .unwrap_or_else(|e| panic!("serving pp512 chunk={chunk} run {run}: {e:#}"));
                last_logits = Some(logits);
            }
            device.synchronize().expect("sync before stopping the clock");
            let dt = t0.elapsed().as_secs_f64();
            assert_eq!(
                cache.current_len(),
                total,
                "serving prefill must commit all {total} tokens"
            );
            let logits = last_logits.expect("at least one chunk ran");
            assert_eq!(
                logits.dims3().expect("logits dims").1,
                1,
                "logit_rows=Some(1) must yield exactly one row"
            );
            let last_row: Vec<f32> = logits
                .to_dtype(DType::F32)
                .expect("f32")
                .flatten_all()
                .expect("flatten")
                .to_vec1()
                .expect("to host");
            let finite = last_row.iter().filter(|v| v.is_finite()).count();
            assert_eq!(
                finite,
                last_row.len(),
                "serving pp512 produced non-finite logits at chunk={chunk}"
            );
            let tok_s = total as f64 / dt;
            let warm = run >= WARMUP_RUNS_1_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW;
            if warm {
                timed_tok_s.push(tok_s);
            }
            eprintln!(
                "PP qwen38-cuda pp512-serving chunk={chunk} run={run} timed={warm} basis=eager_dense_chunked_prefill_lm_head_last_row_per_chunk_fresh_cache tok_s={tok_s:.1} ms={:.1}",
                dt * 1e3
            );
        }
        let mean = timed_tok_s.iter().sum::<f64>() / timed_tok_s.len() as f64;
        assert!(
            mean.is_finite() && mean > 0.0,
            "mean serving pp512 tok/s is not a positive number: {mean}"
        );
        eprintln!(
            "PP qwen38-cuda pp512-serving chunk={chunk} basis=eager_dense_chunked_prefill_lm_head_last_row_per_chunk_fresh_cache runs={} mean_tok_s={mean:.1} run_tok_s={:?}",
            timed_tok_s.len(),
            timed_tok_s
                .iter()
                .map(|v| (v * 10.0).round() / 10.0)
                .collect::<Vec<f64>>()
        );
    }
}
