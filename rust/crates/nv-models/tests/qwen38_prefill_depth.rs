#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{decode_prof, Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::time::Instant;

mod common;
mod ctx_timing_common;
mod hub_snapshot;
use common::qwen38_snapshot_dir_env_override_then_home_hub;

const DEPTHS_DEFAULT_8192_32768_WHERE_THE_VLLM_GAP_IS_WORST: &[usize] = &[8192, 32768];
const WARMUP_RUNS_DEFAULT_1_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW: usize = 1;
const TIMED_RUNS_DEFAULT_2_THE_HONEST_MINIMUM_FOR_A_MACHINE_LINE: usize = 2;
const KV_HEADROOM_8_ROWS_KEEPS_THE_CACHE_STRICTLY_LARGER_THAN_THE_PROMPT: usize = 8;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn depths_from_env() -> Vec<usize> {
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
        Err(_) => DEPTHS_DEFAULT_8192_32768_WHERE_THE_VLLM_GAP_IS_WORST.to_vec(),
    }
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

const BOOT_WARM_AB_DEPTH_8192_CROSSES_ONE_CHUNK_SEAM: usize = 8192;
const BOOT_WARM_AB_LOGIT_ABS_CEILING_1E4_SAME_SHAPES_SAME_KERNELS_ONLY_ALLOCATOR_ADDRESSES_MAY_DIFFER:
    f32 = 1e-4;

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 dense arm TWICE sequentially; set \
            NV_QWEN38_SERVING_TEST=1 -- the boot-warm hygiene contract: an engine built with \
            NV_Q38_BOOT_WARM_PREFILL=1 (one throwaway chunk then reset) must produce the same \
            last-row prefill logits as an engine built without it, because the warm chunk \
            touches only caches the reset clears"]
fn qwen38_cuda_boot_warm_prefill_leaves_no_state_the_reset_missed() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    use nv_models::graph_engine::GraphedQwen3Moe;
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let depth = BOOT_WARM_AB_DEPTH_8192_CROSSES_ONE_CHUNK_SEAM;
    let max_seq = depth + KV_HEADROOM_8_ROWS_KEEPS_THE_CACHE_STRICTLY_LARGER_THAN_THE_PROMPT;
    let prompt: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();

    let mut run_arm = |warm_on: bool| -> Vec<f32> {
        if warm_on {
            std::env::set_var(
                nv_models::graph_engine::NV_Q38_BOOT_WARM_PREFILL_ENV_OPT_IN_1_RUNS_ONE_THROWAWAY_CHUNK_AT_ENGINE_BUILD_BECAUSE_THE_FIRST_REAL_CHUNK_PAYS_5S_OF_ONE_TIME_MODULE_AND_WORKSPACE_INIT_LANDING_IN_TTFT,
                "1",
            );
        } else {
            std::env::remove_var(
                nv_models::graph_engine::NV_Q38_BOOT_WARM_PREFILL_ENV_OPT_IN_1_RUNS_ONE_THROWAWAY_CHUNK_AT_ENGINE_BUILD_BECAUSE_THE_FIRST_REAL_CHUNK_PAYS_5S_OF_ONE_TIME_MODULE_AND_WORKSPACE_INIT_LANDING_IN_TTFT,
            );
        }
        let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
        let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
        let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
        let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
            .expect("dense arm");
        drop(weights);
        let mut engine =
            GraphedQwen3Moe::new(base, &device, max_seq).expect("graphed engine");
        assert_eq!(engine.current_pos(), 0, "boot warm must reset the engine position");
        let row = engine.prefill(&prompt).expect("prefill");
        assert_eq!(engine.current_pos(), depth);
        row
    };

    let row_off = run_arm(false);
    let row_on = run_arm(true);
    std::env::remove_var(
        nv_models::graph_engine::NV_Q38_BOOT_WARM_PREFILL_ENV_OPT_IN_1_RUNS_ONE_THROWAWAY_CHUNK_AT_ENGINE_BUILD_BECAUSE_THE_FIRST_REAL_CHUNK_PAYS_5S_OF_ONE_TIME_MODULE_AND_WORKSPACE_INIT_LANDING_IN_TTFT,
    );
    assert_eq!(row_off.len(), row_on.len(), "arms returned different vocab widths");
    let argmax = |row: &[f32]| -> usize {
        let mut best = 0usize;
        let mut val = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            assert!(v.is_finite(), "non-finite logit at {i}");
            if v > val {
                val = v;
                best = i;
            }
        }
        best
    };
    let max_abs = row_off
        .iter()
        .zip(row_on.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "boot-warm AB depth={depth} argmax_off={} argmax_on={} logit_max_abs={max_abs:.6}",
        argmax(&row_off),
        argmax(&row_on)
    );
    assert_eq!(
        argmax(&row_off),
        argmax(&row_on),
        "boot warm flipped the prefill argmax; the throwaway chunk left state the reset missed"
    );
    assert!(
        max_abs
            <= BOOT_WARM_AB_LOGIT_ABS_CEILING_1E4_SAME_SHAPES_SAME_KERNELS_ONLY_ALLOCATOR_ADDRESSES_MAY_DIFFER,
        "boot warm moved last-row logits by {max_abs} (ceiling {})",
        BOOT_WARM_AB_LOGIT_ABS_CEILING_1E4_SAME_SHAPES_SAME_KERNELS_ONLY_ALLOCATOR_ADDRESSES_MAY_DIFFER
    );
}

const SCAN_V2_AB_DEPTH_8192_TWO_CHUNKS_SO_THE_STATE_CARRY_CROSSES_A_SEAM: usize = 8192;

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 dense arm; set NV_QWEN38_SERVING_TEST=1 -- \
            the NV_Q38_GDN_SCAN_V2 bitwise gate: one engine, two chunked prefills at 8192 \
            (fresh cache each) with the shipped scan kernel then the v2 warp-synchronous \
            kernel, comparing last-row logit BITS -- the v2 kernel replays the identical \
            fp ops in the identical order so 64 layers of downstream compute must not move \
            a single bit"]
fn qwen38_cuda_gdn_scan_v2_prefill_last_row_logits_bitwise_vs_shipped_scan() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    assert_ne!(
        std::env::var("NV_Q38_GDN_CHUNK_PREFILL").ok().as_deref(),
        Some("0"),
        "the chunk scan must be armed or this gate compares the candle path against itself"
    );
    std::env::set_var("NV_Q38_GDN_CHUNK_GEMM", "0");
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("build Qwen3.8-27B on the eager cuda dense arm");
    drop(weights);

    let depth = SCAN_V2_AB_DEPTH_8192_TWO_CHUNKS_SO_THE_STATE_CARRY_CROSSES_A_SEAM;
    let max_seq = depth + KV_HEADROOM_8_ROWS_KEEPS_THE_CACHE_STRICTLY_LARGER_THAN_THE_PROMPT;
    let ids: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
    let chunk = nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
        .filter(|&c| depth > c)
        .unwrap_or(depth);

    let mut run_arm = |v2_on: bool| -> Vec<u32> {
        if v2_on {
            std::env::set_var("NV_Q38_GDN_SCAN_V2", "1");
        } else {
            std::env::remove_var("NV_Q38_GDN_SCAN_V2");
        }
        let mut cache = model.new_kv_cache(max_seq).expect("kv cache");
        let mut fed = 0usize;
        let mut last_logits = None;
        while fed < depth {
            let n = chunk.min(depth - fed);
            let start = cache.current_len();
            let tokens = Tensor::from_vec(ids[fed..fed + n].to_vec(), (1usize, n), &device)
                .expect("tokens");
            let pos = Tensor::from_vec(
                (start as i32..(start + n) as i32).collect::<Vec<i32>>(),
                n,
                &device,
            )
            .expect("positions");
            let logits = model
                .forward_with_cache_dispatched_rows(&tokens, &pos, &mut cache, None, Some(1))
                .unwrap_or_else(|e| panic!("prefill chunk at pos {fed} v2={v2_on}: {e:#}"));
            last_logits = Some(logits);
            fed += n;
        }
        device.synchronize().expect("sync");
        let row: Vec<f32> = last_logits
            .expect("at least one chunk ran")
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to host");
        row.iter().map(|v| v.to_bits()).collect()
    };

    let bits_v1 = run_arm(false);
    let bits_v2 = run_arm(true);
    std::env::remove_var("NV_Q38_GDN_SCAN_V2");
    std::env::remove_var("NV_Q38_GDN_CHUNK_GEMM");
    assert_eq!(bits_v1.len(), bits_v2.len(), "arms returned different vocab widths");
    let diff = bits_v1
        .iter()
        .zip(bits_v2.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff, 0,
        "NV_Q38_GDN_SCAN_V2 moved {diff}/{} last-row logit bits at depth {depth} chunk {chunk}",
        bits_v1.len()
    );
    eprintln!(
        "gdn scan v2 bitwise gate: depth={depth} chunk={chunk} vocab={} all logit bits equal",
        bits_v1.len()
    );
}

const TAILCHAMP_ARMS_ENV_CSV_FROM_SCAN_SCANV2_GEMM_GRAPH_FWDH_MMA_V3: &str = "NV_TAILCHAMP_ARMS";
const TAILCHAMP_ARMS_DEFAULT_MMA_V3_THE_STANDING_CHAMPIONSHIP_PAIR: &str = "mma,v3";
const TAILCHAMP_ROUNDS_ENV: &str = "NV_TAILCHAMP_ROUNDS";
const TAILCHAMP_ROUNDS_DEFAULT_10_BECAUSE_A_SIGN_TEST_NEEDS_9_OF_10_FOR_P_LT_5_PCT: usize = 10;
const TAILCHAMP_WARMUP_ENV: &str = "NV_TAILCHAMP_WARMUP_PASSES";
const TAILCHAMP_WARMUP_DEFAULT_2_PER_ARM_FIRST_PAYS_MODULE_AND_WORKSPACE_INIT_SECOND_RAMPS_CLOCKS:
    usize = 2;
const TAILCHAMP_DEPTH_DEFAULT_8192_ONLY_BECAUSE_EVERY_EXTRA_DEPTH_MULTIPLIES_GPU_MINUTES: &[usize] =
    &[8192];

const GDN_ARM_ENVS_ALL_CLEARED_BEFORE_EACH_SELECTION_BECAUSE_V3_MMA_FWDH_GRAPH_STACK_WITH_PRECEDENCE:
    &[&str] = &[
    "NV_Q38_GDN_SCAN_V2",
    "NV_Q38_GDN_CHUNK_GEMM",
    "NV_Q38_GDN_CHUNK_GEMM_GRAPH",
    "NV_Q38_GDN_CHUNK_GEMM_FWDH",
    "NV_Q38_GDN_CHUNK_GEMM_MMA",
    "NV_Q38_GDN_CHUNK_GEMM_V3",
];

const GDN_DEFAULT_ON_ENVS_PINNED_0_PER_SELECTION_SO_ARM_NAMES_KEEP_THEIR_PRE_FLIP_MEANINGS:
    &[&str] = &["NV_Q38_GDN_CHUNK_GEMM", "NV_Q38_GDN_CHUNK_GEMM_V3"];

fn select_gdn_arm_envs_reread_per_call_by_both_the_rust_and_cu_sides(arm: &str) {
    for e in GDN_ARM_ENVS_ALL_CLEARED_BEFORE_EACH_SELECTION_BECAUSE_V3_MMA_FWDH_GRAPH_STACK_WITH_PRECEDENCE
    {
        std::env::remove_var(e);
    }
    for e in GDN_DEFAULT_ON_ENVS_PINNED_0_PER_SELECTION_SO_ARM_NAMES_KEEP_THEIR_PRE_FLIP_MEANINGS {
        std::env::set_var(e, "0");
    }
    let set = |envs: &[&str]| {
        for e in envs {
            std::env::set_var(e, "1");
        }
    };
    match arm {
        "scan" => {}
        "scanv2" => set(&["NV_Q38_GDN_SCAN_V2"]),
        "gemm" => set(&["NV_Q38_GDN_CHUNK_GEMM"]),
        "graph" => set(&["NV_Q38_GDN_CHUNK_GEMM", "NV_Q38_GDN_CHUNK_GEMM_GRAPH"]),
        "fwdh" => set(&["NV_Q38_GDN_CHUNK_GEMM", "NV_Q38_GDN_CHUNK_GEMM_FWDH"]),
        "mma" => set(&["NV_Q38_GDN_CHUNK_GEMM", "NV_Q38_GDN_CHUNK_GEMM_MMA"]),
        "v3" => set(&["NV_Q38_GDN_CHUNK_GEMM", "NV_Q38_GDN_CHUNK_GEMM_V3"]),
        other => panic!(
            "NV_TAILCHAMP_ARMS entry {other:?} unknown: scan|scanv2|gemm|graph|fwdh|mma|v3"
        ),
    }
}

fn round_order_thue_morse_abba_baab_so_linear_clock_drift_cancels_in_adjacent_pair_blocks(
    round: usize,
    n_arms: usize,
) -> Vec<usize> {
    let forward = (round.count_ones() % 2) == 0;
    if forward {
        (0..n_arms).collect()
    } else {
        (0..n_arms).rev().collect()
    }
}

fn median_f64(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.len() % 2 == 1 {
        s[s.len() / 2]
    } else {
        (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2.0
    }
}

fn sign_test_two_sided_exact_binomial_p(wins: usize, n: usize) -> f64 {
    let k = wins.min(n - wins);
    let mut tail = 0.0f64;
    let mut coeff = 1.0f64;
    for i in 0..=k {
        if i > 0 {
            coeff = coeff * (n - i + 1) as f64 / i as f64;
        }
        tail += coeff * 0.5f64.powi(n as i32);
    }
    (2.0 * tail).min(1.0)
}

fn sample_sm_clock_mhz_via_nvidia_smi_or_zero_when_unavailable() -> u32 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .and_then(|l| l.trim().parse::<u32>().ok())
        })
        .unwrap_or(0)
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 dense arm; set NV_QWEN38_SERVING_TEST=1 -- the \
            suite covering in-process GDN arm championships: ONE process, real weights, the \
            serving chunk loop (fresh cache per pass, NV_Q38_PREFILL_CHUNK default 4096, lm_head \
            last row), arms env-selected PER PASS from NV_TAILCHAMP_ARMS (default mma,v3; also \
            scan|scanv2|gemm|graph|fwdh) because linear_attn.rs and gdn_chunk_gemm.cu re-read \
            their arm envs on every call; NV_TAILCHAMP_WARMUP_PASSES per arm (default 2) then \
            NV_TAILCHAMP_ROUNDS rounds (default 10) of one pass per arm in Thue-Morse mirrored \
            order (ABBA BAAB ...), per-pass wall + sampled SM clock; verdicts are per-arm-pair \
            PAIRED per-round deltas -- median delta pct plus an exact two-sided binomial sign \
            test -- because the cross-process A/B protocol's ~1 % order/boot noise floor \
            swallowed the mma-vs-v3 question; depths via NV_CTX_TOKENS (default 8192 only)"]
fn qwen38_cuda_gdn_arm_championship_in_process_alternating_prefill_sign_test() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let arms: Vec<String> = std::env::var(TAILCHAMP_ARMS_ENV_CSV_FROM_SCAN_SCANV2_GEMM_GRAPH_FWDH_MMA_V3)
        .unwrap_or_else(|_| TAILCHAMP_ARMS_DEFAULT_MMA_V3_THE_STANDING_CHAMPIONSHIP_PAIR.into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    assert!(arms.len() >= 2, "a championship needs at least two arms");
    let rounds = env_usize(
        TAILCHAMP_ROUNDS_ENV,
        TAILCHAMP_ROUNDS_DEFAULT_10_BECAUSE_A_SIGN_TEST_NEEDS_9_OF_10_FOR_P_LT_5_PCT,
    );
    assert!(rounds >= 1, "NV_TAILCHAMP_ROUNDS must be >= 1");
    let warmup = env_usize(
        TAILCHAMP_WARMUP_ENV,
        TAILCHAMP_WARMUP_DEFAULT_2_PER_ARM_FIRST_PAYS_MODULE_AND_WORKSPACE_INIT_SECOND_RAMPS_CLOCKS,
    );
    let depths = match std::env::var("NV_CTX_TOKENS") {
        Ok(_) => depths_from_env(),
        Err(_) => TAILCHAMP_DEPTH_DEFAULT_8192_ONLY_BECAUSE_EVERY_EXTRA_DEPTH_MULTIPLIES_GPU_MINUTES
            .to_vec(),
    };

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let rev = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let n_layers = cfg.num_hidden_layers;
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("build Qwen3.8-27B on the eager cuda dense arm");
    drop(weights);

    for &depth in &depths {
        let max_seq = depth + KV_HEADROOM_8_ROWS_KEEPS_THE_CACHE_STRICTLY_LARGER_THAN_THE_PROMPT;
        let ids: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        let chunk = nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
            .filter(|&c| depth > c)
            .unwrap_or(depth);

        let one_pass = |arm: &str| -> f64 {
            select_gdn_arm_envs_reread_per_call_by_both_the_rust_and_cu_sides(arm);
            let mut cache = model.new_kv_cache(max_seq).expect("kv cache");
            device.synchronize().expect("sync before starting the clock");
            let t0 = Instant::now();
            let mut fed = 0usize;
            while fed < depth {
                let n = chunk.min(depth - fed);
                let start = cache.current_len();
                let tokens = Tensor::from_vec(ids[fed..fed + n].to_vec(), (1usize, n), &device)
                    .expect("tokens");
                let pos = Tensor::from_vec(
                    (start as i32..(start + n) as i32).collect::<Vec<i32>>(),
                    n,
                    &device,
                )
                .expect("positions");
                let logits = model
                    .forward_with_cache_dispatched_rows(&tokens, &pos, &mut cache, None, Some(1))
                    .unwrap_or_else(|e| panic!("prefill chunk at pos {fed} arm {arm}: {e:#}"));
                drop(logits);
                fed += n;
            }
            device.synchronize().expect("sync before stopping the clock");
            let dt = t0.elapsed().as_secs_f64();
            assert_eq!(cache.current_len(), depth, "prefill must commit all {depth} tokens");
            dt
        };

        for arm in &arms {
            for w in 0..warmup {
                let dt = one_pass(arm);
                eprintln!(
                    "tailchamp warmup depth={depth} arm={arm} pass={w} wall_s={dt:.4} \
                     sm_mhz={}",
                    sample_sm_clock_mhz_via_nvidia_smi_or_zero_when_unavailable()
                );
            }
        }

        let mut walls: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        let mut clocks: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for round in 0..rounds {
            let order =
                round_order_thue_morse_abba_baab_so_linear_clock_drift_cancels_in_adjacent_pair_blocks(
                    round,
                    arms.len(),
                );
            for &ai in &order {
                let dt = one_pass(&arms[ai]);
                let mhz = sample_sm_clock_mhz_via_nvidia_smi_or_zero_when_unavailable();
                walls[ai].push(dt);
                clocks[ai].push(mhz as f64);
                eprintln!(
                    "tailchamp pass depth={depth} round={round} arm={} wall_s={dt:.4} \
                     tok_s={:.1} sm_mhz={mhz}",
                    arms[ai],
                    depth as f64 / dt
                );
            }
        }
        for e in GDN_ARM_ENVS_ALL_CLEARED_BEFORE_EACH_SELECTION_BECAUSE_V3_MMA_FWDH_GRAPH_STACK_WITH_PRECEDENCE
        {
            std::env::remove_var(e);
        }

        for (ai, arm) in arms.iter().enumerate() {
            assert_eq!(walls[ai].len(), rounds, "arm {arm} must run once per round");
            let med = median_f64(&walls[ai]);
            println!(
                "QWEN38-CUDA-PREFILL checkpoint=unsloth/Qwen3.8-27B-NVFP4@{rev} backend=cuda \
                 layers={n_layers} prompt_tokens={depth} prefill_s={med:.4} tok_s={:.1} \
                 chunk={chunk} arm={arm} passes={rounds} \
                 pass_basis=median_of_in_process_thue_morse_alternation sm_mhz_median={:.0}",
                depth as f64 / med,
                median_f64(&clocks[ai])
            );
        }
        for ai in 0..arms.len() {
            for bi in ai + 1..arms.len() {
                let deltas_pct: Vec<f64> = (0..rounds)
                    .map(|r| (walls[ai][r] - walls[bi][r]) / walls[ai][r] * 100.0)
                    .collect();
                let wins_b = deltas_pct.iter().filter(|&&d| d > 0.0).count();
                let ties = deltas_pct.iter().filter(|&&d| d == 0.0).count();
                let p = sign_test_two_sided_exact_binomial_p(wins_b, rounds - ties);
                println!(
                    "tailchamp pair depth={depth} a={} b={} median_a_s={:.4} median_b_s={:.4} \
                     paired_median_delta_pct={:.3} b_wins={wins_b}/{} sign_p={p:.4} \
                     method=paired_per_round_exact_two_sided_binomial_sign_test",
                    arms[ai],
                    arms[bi],
                    median_f64(&walls[ai]),
                    median_f64(&walls[bi]),
                    median_f64(&deltas_pct),
                    rounds - ties
                );
            }
        }
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 dense arm; set NV_QWEN38_SERVING_TEST=1 -- \
            serving-semantic chunked prefill wall vs prompt depth: the exact chunk loop \
            graph_engine::prefill runs (NV_Q38_PREFILL_CHUNK default 4096, lm_head last row per \
            chunk, fresh cache per run), depths via NV_CTX_TOKENS (default 8192,32768), runs via \
            NV_PREFILL_WARMUP_RUNS/NV_PREFILL_TIMED_RUNS; NV_PROF_PREFILL=1 adds per-chunk mixer \
            wall splits and NV_PROF_DECODE=1 arms the synced per-op profiler per chunk -- \
            profiled absolutes never enter the store"]
fn qwen38_cuda_chunked_prefill_tok_s_vs_prompt_depth_serving_last_row() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let rev = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let n_layers = cfg.num_hidden_layers;
    let max_pos = cfg.max_position_embeddings;
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("build Qwen3.8-27B on the eager cuda dense arm");
    drop(weights);
    assert!(model.is_dense(), "from_loader_dense_quantized must yield the dense arm");

    let warmup = env_usize(
        "NV_PREFILL_WARMUP_RUNS",
        WARMUP_RUNS_DEFAULT_1_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW,
    );
    let timed = env_usize(
        "NV_PREFILL_TIMED_RUNS",
        TIMED_RUNS_DEFAULT_2_THE_HONEST_MINIMUM_FOR_A_MACHINE_LINE,
    );
    assert!(timed >= 1, "NV_PREFILL_TIMED_RUNS must be >= 1");
    let prof_armed = std::env::var("NV_PROF_DECODE").ok().as_deref() == Some("1");

    for depth in depths_from_env() {
        let max_seq = depth + KV_HEADROOM_8_ROWS_KEEPS_THE_CACHE_STRICTLY_LARGER_THAN_THE_PROMPT;
        assert!(
            max_seq < max_pos,
            "a model declaring max_pos {max_pos} must serve depth {depth}"
        );
        let ids: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        let chunk = nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
            .filter(|&c| depth > c)
            .unwrap_or(depth);
        for run in 0..warmup + timed {
            let mut cache = model.new_kv_cache(max_seq).expect("kv cache");
            device.synchronize().expect("sync before starting the clock");
            let t0 = Instant::now();
            let mut fed = 0usize;
            let mut last_logits = None;
            while fed < depth {
                let n = chunk.min(depth - fed);
                let start = cache.current_len();
                let tokens =
                    Tensor::from_vec(ids[fed..fed + n].to_vec(), (1usize, n), &device)
                        .expect("tokens");
                let pos = Tensor::from_vec(
                    (start as i32..(start + n) as i32).collect::<Vec<i32>>(),
                    n,
                    &device,
                )
                .expect("positions");
                if prof_armed {
                    decode_prof::begin_env_nv_prof_decode_refusing_mid_capture_because_every_lap_syncs(&device);
                }
                let logits = model
                    .forward_with_cache_dispatched_rows(&tokens, &pos, &mut cache, None, Some(1))
                    .unwrap_or_else(|e| panic!("prefill chunk at pos {fed} depth {depth}: {e:#}"));
                last_logits = Some(logits);
                fed += n;
            }
            device.synchronize().expect("sync before stopping the clock");
            let dt = t0.elapsed().as_secs_f64();
            assert_eq!(cache.current_len(), depth, "prefill must commit all {depth} tokens");
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
                "prefill produced non-finite logits at depth {depth}"
            );
            let tok_s = depth as f64 / dt;
            let warm = run >= warmup;
            if warm {
                println!(
                    "QWEN38-CUDA-PREFILL checkpoint=unsloth/Qwen3.8-27B-NVFP4@{rev} backend=cuda \
                     layers={n_layers} prompt_tokens={depth} prefill_s={dt:.2} tok_s={tok_s:.1} \
                     chunk={chunk} run={run}"
                );
            } else {
                eprintln!(
                    "qwen38-cuda-prefill warmup depth={depth} run={run} prefill_s={dt:.2} \
                     tok_s={tok_s:.1} chunk={chunk}"
                );
            }
        }
    }
}
