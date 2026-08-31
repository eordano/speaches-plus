#![cfg(feature = "cuda")]

mod ctx_timing_common;
mod hub_snapshot;

use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b::Gemma4E4b;
use nv_weights::WeightLoader;
use std::time::Instant;

const QAT_REPO: &str = "google/gemma-4-E4B-it-qat-w4a16-ct";
const BF16_REPO: &str = "google/gemma-4-E4B-it";
const SERVING_PREFILL_CHUNK_512_MATCHES_RUN_SAMPLING_GEMMA4_E4B: usize = 512;
const SERVING_GRAPH_BATCH_16_MATCHES_RUN_SAMPLING_GEMMA4_E4B: usize = 16;
const SESSION_ROUNDS_2_THE_FIRST_IS_THE_COLD_PROCESS_THE_SECOND_IS_A_WARM_SERVER_REQUEST: usize =
    2;
const TIMED_REPLAY_BATCHES_8_GIVE_128_TOKENS_FOR_A_STABLE_MEDIAN: usize = 8;
const KV_HEADROOM_1024_MATCHES_THE_CTX_DEPTH_LADDER: usize = 1024;

fn ctx_depths_from_env_one_grammar_for_every_test_in_this_binary_comma_list_each_entry_optional_k_suffix(
    default: &[usize],
) -> Vec<usize> {
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

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
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

fn serving_checkpoint_dir() -> (std::path::PathBuf, String, &'static str) {
    let dir = hub_snapshot::dir_from_env_or_hub(
        "NV_E4B_DIR",
        QAT_REPO,
        &["config.json", "*.safetensors"],
    )
    .or_else(|| {
        hub_snapshot::dir_from_env_or_hub("NV_E4B_DIR", BF16_REPO, &["config.json", "*.safetensors"])
    })
    .expect("neither E4B snapshot is hydrated; cache the qat-w4a16-ct or bf16 checkpoint");
    let basis = basis_read_from_config_json_quantization_config_not_from_probe_order_because_nv_e4b_dir_may_hold_either_checkpoint(&dir);
    let repo = if basis == "w4a16-qat" { QAT_REPO } else { BF16_REPO };
    let rev = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());
    (dir, format!("{repo}@{rev}"), basis)
}

fn load_model(dir: &std::path::Path, device: &Device) -> (Gemma4E4b, f64) {
    let cfg = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    assert!(cfg.has_per_layer_embeddings(), "not an E4B checkpoint: {dir:?}");
    let weights = WeightLoader::open_dir(dir, device).expect("weights");
    let t0 = Instant::now();
    let model = Gemma4E4b::from_loader(cfg, &weights, device).expect("load Gemma4E4b");
    device.synchronize().expect("sync after load");
    (model, t0.elapsed().as_secs_f64())
}

fn argmax_last_row(logits: &Tensor) -> u32 {
    let row: Vec<f32> = logits
        .to_dtype(DType::F32)
        .expect("f32")
        .to_vec1()
        .expect("host row");
    let mut best = 0usize;
    let mut val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        assert!(v.is_finite(), "non-finite logit at {i}");
        if v > val {
            val = v;
            best = i;
        }
    }
    best as u32
}

#[test]
#[ignore = "loads the gemma-4 E4B serving checkpoint (qat-w4a16-ct preferred, bf16 fallback -- \
            the ckpt= field says which); set NV_E4B_SERVING_TEST=1 -- walls each phase of the \
            cuda serving session start exactly as run_sampling_gemma4_e4b runs it (chunk-512 \
            eager prefill into a cat cache, graphed_decoder build, warm_step, first \
            replay_batch capture, replayed decode) at the first NV_CTX_TOKENS entry (comma \
            list, k suffix ok, default 8192), \
            twice in one process: round 0 carries every one-time cold cost, round 1 is what a \
            warm server pays per request"]
fn e4b_cuda_serving_session_phase_walls_cold_then_warm() {
    require_gate("NV_E4B_SERVING_TEST");
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let (dir, ckpt, basis) = serving_checkpoint_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let depth = ctx_depths_from_env_one_grammar_for_every_test_in_this_binary_comma_list_each_entry_optional_k_suffix(&[8192])[0];

    let (model, build_s) = load_model(&dir, &device);
    let max_pos = model.config().max_position_embeddings;
    assert!(depth + KV_HEADROOM_1024_MATCHES_THE_CTX_DEPTH_LADDER < max_pos);
    eprintln!("e4b checkpoint: {ckpt} ({basis}) model_build_s={build_s:.3}");

    let prompt: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
    let n_layers = model.config().num_hidden_layers;
    let graph_batch = SERVING_GRAPH_BATCH_16_MATCHES_RUN_SAMPLING_GEMMA4_E4B;
    let decode_budget = (1 + TIMED_REPLAY_BATCHES_8_GIVE_128_TOKENS_FOR_A_STABLE_MEDIAN)
        * graph_batch
        + 1;

    for round in 0..SESSION_ROUNDS_2_THE_FIRST_IS_THE_COLD_PROCESS_THE_SECOND_IS_A_WARM_SERVER_REQUEST
    {
        if let Err(e) = device.synchronize() {
            eprintln!(
                "drained the one-shot GraphedE4bDecoder teardown error before round {round}: {e:?}"
            );
        }
        let t_all = Instant::now();
        let t0 = Instant::now();
        let mut cache: Vec<Option<(Tensor, Tensor)>> = vec![None; n_layers];
        device.synchronize().expect("sync");
        let setup_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let mut off = 0usize;
        let mut last_logits = None;
        while off < prompt.len() {
            let c = (prompt.len() - off)
                .min(SERVING_PREFILL_CHUNK_512_MATCHES_RUN_SAMPLING_GEMMA4_E4B);
            last_logits = Some(
                model
                    .forward_step(&prompt[off..off + c], off, &mut cache)
                    .unwrap_or_else(|e| panic!("prefill chunk at {off}: {e:#}")),
            );
            off += c;
        }
        device.synchronize().expect("sync");
        let trunk_prefill_s = t0.elapsed().as_secs_f64();
        let seed = argmax_last_row(&last_logits.expect("non-empty prompt"));

        let t0 = Instant::now();
        let max_len = (prompt.len() + decode_budget + graph_batch + 8)
            .next_power_of_two()
            .max(64);
        let mut dec = model
            .graphed_decoder(&cache, prompt.len(), max_len)
            .expect("graphed decoder");
        device.synchronize().expect("sync");
        let graph_build_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let mut tok = dec.warm_step(seed).expect("warm step");
        device.synchronize().expect("sync");
        let warm_step_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let batch = dec.replay_batch(tok, graph_batch).expect("capture replay");
        tok = *batch.last().expect("non-empty batch");
        device.synchronize().expect("sync");
        let capture_replay_s = t0.elapsed().as_secs_f64();
        let total_s = t_all.elapsed().as_secs_f64();

        let mut batch_ms: Vec<f64> = Vec::new();
        for _ in 0..TIMED_REPLAY_BATCHES_8_GIVE_128_TOKENS_FOR_A_STABLE_MEDIAN {
            let t0 = Instant::now();
            let batch = dec.replay_batch(tok, graph_batch).expect("timed replay");
            tok = *batch.last().expect("non-empty batch");
            device.synchronize().expect("sync");
            batch_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        batch_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms_tok = batch_ms[batch_ms.len() / 2] / graph_batch as f64;
        let steps = TIMED_REPLAY_BATCHES_8_GIVE_128_TOKENS_FOR_A_STABLE_MEDIAN * graph_batch;

        eprintln!(
            "PREFILL-PHASES-E4B depth={depth} round={round} ckpt={ckpt} seed={seed} \
             setup_s={setup_s:.3} trunk_prefill_s={trunk_prefill_s:.3} \
             graph_build_s={graph_build_s:.3} warm_step_s={warm_step_s:.3} \
             capture_replay_s={capture_replay_s:.3} total_s={total_s:.3} \
             trunk_tok_s={:.1} decode_median_ms_tok={median_ms_tok:.3}",
            depth as f64 / trunk_prefill_s,
        );
        if round == 0 {
            println!(
                "CTX-SCALING e4b-cuda-serving depth={depth} arm=graphed steps={steps} \
                 warmup_steps={} median_ms_tok={median_ms_tok:.3} tok_s={:.1} \
                 prefill_s={trunk_prefill_s:.2} engine_build_s={build_s:.2} \
                 capture_round0_ms={:.1} checkpoint={ckpt}",
                graph_batch + 1,
                1000.0 / median_ms_tok,
                (graph_build_s + warm_step_s + capture_replay_s) * 1e3,
            );
        }
    }
}

const BOOT_WARM_AB_DEPTH_8192_CROSSES_MANY_CHUNK_SEAMS: usize = 8192;
const BOOT_WARM_AB_LOGIT_ABS_FLOOR_1E4_WHEN_THE_REBUILD_CONTROL_IS_EXACT: f32 = 1e-4;
const BOOT_WARM_AB_CONTROL_MARGIN_2X_BOUNDS_THE_ARM_BY_THE_MEASURED_FRESH_PROCESS_REBUILD_NOISE_BECAUSE_E4B_W4A16_PREFILL_LOGITS_CARRY_ATOMIC_ACCUMULATION_NOISE_ON_REBUILD:
    f32 = 2.0;

const BOOT_WARM_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM: &str =
    "NV_E4B_BOOT_WARM_AB_CHILD_ROW_OUT";

fn ab_child_writes_last_prefill_row_le_f32(row_out: &str, prompt: &[u32]) {
    let (dir, ckpt, _basis) = serving_checkpoint_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let (model, _build_s) = load_model(&dir, &device);
    let mut cache: Vec<Option<(Tensor, Tensor)>> = vec![None; model.config().num_hidden_layers];
    let mut off = 0usize;
    let mut last_logits = None;
    while off < prompt.len() {
        let c = (prompt.len() - off).min(SERVING_PREFILL_CHUNK_512_MATCHES_RUN_SAMPLING_GEMMA4_E4B);
        last_logits = Some(
            model
                .forward_step(&prompt[off..off + c], off, &mut cache)
                .unwrap_or_else(|e| panic!("prefill chunk at {off}: {e:#}")),
        );
        off += c;
    }
    let row: Vec<f32> = last_logits
        .expect("non-empty prompt")
        .to_dtype(DType::F32)
        .expect("f32")
        .to_vec1()
        .expect("host row");
    let mut bytes = Vec::with_capacity(row.len() * 4);
    for v in &row {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(row_out, bytes).expect("write child row");
    eprintln!("boot-warm AB child ckpt={ckpt} vocab={}", row.len());
}

#[test]
#[ignore = "re-execs the test binary so EACH arm loads the gemma-4 E4B serving checkpoint in a \
            FRESH process, because the warm chunk's production configuration is the first GPU \
            work of a cold process and an in-process A/B cannot see state seeded \
            only-when-first; set NV_E4B_SERVING_TEST=1 -- the boot-warm hygiene contract: a \
            model built with NV_E4B_BOOT_WARM_PREFILL=1 (one throwaway chunk into a scratch \
            cache dropped before return) must keep the prefill argmax and stay within the \
            measured fresh-process rebuild-noise ceiling of a model built without it (E4B \
            W4A16 prefill logits carry atomic accumulation noise across rebuilds, so exact \
            equality is not the property this path has)"]
fn e4b_cuda_boot_warm_prefill_leaves_no_state_a_request_can_see() {
    require_gate("NV_E4B_SERVING_TEST");
    let depth = BOOT_WARM_AB_DEPTH_8192_CROSSES_MANY_CHUNK_SEAMS;
    let prompt: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
    if let Ok(row_out) =
        std::env::var(BOOT_WARM_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM)
    {
        ab_child_writes_last_prefill_row_le_f32(&row_out, &prompt);
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let (_dir, ckpt, _basis) = serving_checkpoint_dir();

    let run_arm = |warm_on: bool, tag: &str| -> Vec<f32> {
        let exe = std::env::current_exe().expect("test binary path");
        let row_out = std::env::temp_dir().join(format!(
            "e4b-boot-warm-ab-{}-{tag}.f32row",
            std::process::id()
        ));
        let mut cmd = std::process::Command::new(exe);
        cmd.args([
            "--ignored",
            "--exact",
            "e4b_cuda_boot_warm_prefill_leaves_no_state_a_request_can_see",
            "--nocapture",
        ]);
        cmd.env(
            BOOT_WARM_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM,
            &row_out,
        );
        let warm_env = nv_models::gemma4_e4b::NV_E4B_BOOT_WARM_PREFILL_ENV_OPT_IN_1_RUNS_ONE_THROWAWAY_CHUNK_AT_MODEL_BUILD_BECAUSE_THE_FIRST_REAL_CHUNK_OF_A_FRESH_PROCESS_PAYS_ONE_TIME_MODULE_AND_WORKSPACE_INIT_LANDING_IN_TTFT;
        if warm_on {
            cmd.env(warm_env, "1");
        } else {
            cmd.env_remove(warm_env);
        }
        let out = cmd.output().expect("spawn fresh-process arm");
        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        assert!(
            out.status.success(),
            "fresh-process arm {tag} failed:\n{}",
            String::from_utf8_lossy(&out.stdout)
        );
        let bytes = std::fs::read(&row_out)
            .unwrap_or_else(|e| panic!("child row file {}: {e}", row_out.display()));
        std::fs::remove_file(&row_out).ok();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let row_off = run_arm(false, "off");
    let row_off_control = run_arm(false, "off-control");
    let row_on = run_arm(true, "on");
    assert_eq!(row_off.len(), row_on.len(), "arms returned different vocab widths");
    let control_max_abs = row_off
        .iter()
        .zip(row_off_control.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("boot-warm AB off-vs-off rebuild control logit_max_abs={control_max_abs:.6}");
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
        "boot-warm AB ckpt={ckpt} depth={depth} argmax_off={} argmax_on={} logit_max_abs={max_abs:.6}",
        argmax(&row_off),
        argmax(&row_on)
    );
    assert_eq!(
        argmax(&row_off),
        argmax(&row_on),
        "boot warm flipped the prefill argmax; the throwaway chunk left visible state"
    );
    let ceiling = (control_max_abs
        * BOOT_WARM_AB_CONTROL_MARGIN_2X_BOUNDS_THE_ARM_BY_THE_MEASURED_FRESH_PROCESS_REBUILD_NOISE_BECAUSE_E4B_W4A16_PREFILL_LOGITS_CARRY_ATOMIC_ACCUMULATION_NOISE_ON_REBUILD)
        .max(BOOT_WARM_AB_LOGIT_ABS_FLOOR_1E4_WHEN_THE_REBUILD_CONTROL_IS_EXACT);
    assert!(
        max_abs <= ceiling,
        "boot warm moved last-row logits by {max_abs}, above the rebuild-noise ceiling {ceiling} \
         (off-vs-off control {control_max_abs})"
    );
}

const SWNARROW_AB_DEPTHS_8192_MATCHES_THE_BOOT_WARM_GATE_AND_1800_ENDS_ON_A_PARTIAL_264_CHUNK_BOTH_CROSS_THE_512_SLIDING_WINDOW:
    [usize; 2] = [8192, 1800];
const SWNARROW_AB_LOGIT_ABS_FLOOR_1E4_WHEN_THE_REBUILD_CONTROL_IS_EXACT: f32 = 1e-4;
const SWNARROW_AB_CONTROL_MARGIN_2X_BOUNDS_THE_ARM_BY_THE_MEASURED_FRESH_PROCESS_REBUILD_NOISE_BECAUSE_E4B_W4A16_PREFILL_LOGITS_CARRY_ATOMIC_ACCUMULATION_NOISE_ON_REBUILD:
    f32 = 2.0;
const SWNARROW_AB_OFF_REBUILDS_3_BECAUSE_ONE_CONTROL_PAIR_UNDERSTATES_THE_REBUILD_NOISE_SPREAD_MEASURED_0_70_TO_2_36_ACROSS_OFF_PAIRS_AT_DEPTH_1800: usize =
    3;
const SWNARROW_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM: &str =
    "NV_E4B_SWNARROW_AB_CHILD_ROW_OUT";
const SWNARROW_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC: &str =
    "NV_E4B_SWNARROW_AB_CHILD_DEPTH";

#[test]
#[ignore = "re-execs the test binary so EACH arm loads the gemma-4 E4B serving checkpoint in a \
            FRESH process (same structure as the boot-warm gate); set NV_E4B_SERVING_TEST=1 -- \
            the sliding-window K/V narrowing identity contract: a chunk-512 prefill with \
            NV_E4B_PREFILL_SLIDING_NARROW=1 (the 35 sliding layers score only the last \
            window+chunk KV rows under a shifted mask) must keep the full-width prefill argmax \
            and stay within 2x the measured off-vs-off fresh-process rebuild noise, at depths \
            8192 (many whole-chunk seams) and 1800 (partial final chunk), both past the \
            512-token sliding window where the narrowing actually drops columns (E4B W4A16 \
            prefill logits carry atomic accumulation noise across rebuilds, so exact equality \
            is not the property this path has)"]
fn e4b_cuda_prefill_sliding_narrow_keeps_argmax_and_stays_within_rebuild_noise_of_full_width() {
    require_gate("NV_E4B_SERVING_TEST");
    if let Ok(row_out) = std::env::var(SWNARROW_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM)
    {
        let depth: usize = std::env::var(
            SWNARROW_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC,
        )
        .expect("child depth env")
        .parse()
        .expect("child depth parse");
        let prompt: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        ab_child_writes_last_prefill_row_le_f32(&row_out, &prompt);
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let (_dir, ckpt, _basis) = serving_checkpoint_dir();

    for depth in
        SWNARROW_AB_DEPTHS_8192_MATCHES_THE_BOOT_WARM_GATE_AND_1800_ENDS_ON_A_PARTIAL_264_CHUNK_BOTH_CROSS_THE_512_SLIDING_WINDOW
    {
        let run_arm = |narrow_on: bool, tag: &str| -> Vec<f32> {
            let exe = std::env::current_exe().expect("test binary path");
            let row_out = std::env::temp_dir().join(format!(
                "e4b-swnarrow-ab-{}-{depth}-{tag}.f32row",
                std::process::id()
            ));
            let mut cmd = std::process::Command::new(exe);
            cmd.args([
                "--ignored",
                "--exact",
                "e4b_cuda_prefill_sliding_narrow_keeps_argmax_and_stays_within_rebuild_noise_of_full_width",
                "--nocapture",
            ]);
            cmd.env(
                SWNARROW_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM,
                &row_out,
            );
            cmd.env(
                SWNARROW_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC,
                depth.to_string(),
            );
            let narrow_env = nv_models::gemma4_e4b::NV_E4B_PREFILL_SLIDING_NARROW_ENV_OPT_IN_1_SCORES_THE_35_SLIDING_LAYERS_AGAINST_ONLY_THE_LAST_WINDOW_512_PLUS_CHUNK_KV_ROWS_WITH_A_SHIFTED_MASK_BECAUSE_THE_SLIDING_MASK_ALREADY_DISCARDS_EVERY_OLDER_COLUMN_SO_FULL_HISTORY_SCORING_IS_PURE_WASTE_KV_SHARING_NEVER_CROSSES_LAYER_TYPES_AND_THE_CACHE_ITSELF_STAYS_FULL_HISTORY_FOR_DECODE;
            if narrow_on {
                cmd.env(narrow_env, "1");
            } else {
                cmd.env_remove(narrow_env);
            }
            let out = cmd.output().expect("spawn fresh-process arm");
            eprintln!("{}", String::from_utf8_lossy(&out.stderr));
            assert!(
                out.status.success(),
                "fresh-process arm {tag} at depth {depth} failed:\n{}",
                String::from_utf8_lossy(&out.stdout)
            );
            let bytes = std::fs::read(&row_out)
                .unwrap_or_else(|e| panic!("child row file {}: {e}", row_out.display()));
            std::fs::remove_file(&row_out).ok();
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        let rows_off: Vec<Vec<f32>> = (0
            ..SWNARROW_AB_OFF_REBUILDS_3_BECAUSE_ONE_CONTROL_PAIR_UNDERSTATES_THE_REBUILD_NOISE_SPREAD_MEASURED_0_70_TO_2_36_ACROSS_OFF_PAIRS_AT_DEPTH_1800)
            .map(|i| run_arm(false, &format!("off{i}")))
            .collect();
        let row_on = run_arm(true, "on");
        assert_eq!(
            rows_off[0].len(),
            row_on.len(),
            "arms returned different vocab widths at depth {depth}"
        );
        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
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
        let mut control = 0f32;
        for i in 0..rows_off.len() {
            for j in i + 1..rows_off.len() {
                control = control.max(max_abs(&rows_off[i], &rows_off[j]));
            }
        }
        let arm = rows_off
            .iter()
            .map(|off| max_abs(off, &row_on))
            .fold(0f32, f32::max);
        eprintln!(
            "swnarrow AB ckpt={ckpt} depth={depth} argmax_off={} argmax_on={} \
             control_logit_max_abs={control:.6} arm_logit_max_abs={arm:.6}",
            argmax(&rows_off[0]),
            argmax(&row_on)
        );
        for off in &rows_off {
            assert_eq!(
                argmax(off),
                argmax(&rows_off[0]),
                "off rebuilds disagree on argmax at depth {depth}: the control itself is unstable"
            );
        }
        assert_eq!(
            argmax(&rows_off[0]),
            argmax(&row_on),
            "sliding-narrow prefill flipped the argmax at depth {depth}"
        );
        let ceiling = (control
            * SWNARROW_AB_CONTROL_MARGIN_2X_BOUNDS_THE_ARM_BY_THE_MEASURED_FRESH_PROCESS_REBUILD_NOISE_BECAUSE_E4B_W4A16_PREFILL_LOGITS_CARRY_ATOMIC_ACCUMULATION_NOISE_ON_REBUILD)
            .max(SWNARROW_AB_LOGIT_ABS_FLOOR_1E4_WHEN_THE_REBUILD_CONTROL_IS_EXACT);
        assert!(
            arm <= ceiling,
            "sliding-narrow prefill moved last-row logits by {arm} at depth {depth}, above the \
             rebuild-noise ceiling {ceiling} (off-vs-off control {control})"
        );
    }
}

const PFFLASH_MASK_TRANSFER_TOL_5E2_SEPARATES_FLASH_ACCUM_ROUNDING_FROM_ORDER_ONE_WRONG_KEY_MASS:
    f32 = 5e-2;

fn eager_reference_on_the_same_f16_rounded_inputs(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n: usize,
    past: usize,
    total: usize,
    kept: usize,
    window: Option<usize>,
    nh: usize,
    nkv: usize,
    hd: usize,
    device: &Device,
) -> Vec<f32> {
    let group = nh / nkv;
    let qf = q.to_dtype(DType::F32).unwrap();
    let kf = k.to_dtype(DType::F32).unwrap();
    let vf = v.to_dtype(DType::F32).unwrap();
    let expand = |t: &Tensor| -> Tensor {
        t.squeeze(0)
            .unwrap()
            .transpose(0, 1)
            .unwrap()
            .unsqueeze(1)
            .unwrap()
            .expand((nkv, group, kept, hd))
            .unwrap()
            .reshape((nh, kept, hd))
            .unwrap()
            .contiguous()
            .unwrap()
    };
    let qh = qf
        .squeeze(0)
        .unwrap()
        .transpose(0, 1)
        .unwrap()
        .contiguous()
        .unwrap();
    let kh = expand(&kf);
    let vh = expand(&vf);
    let off = total - kept;
    let mut mask = vec![0f32; n * kept];
    for r in 0..n {
        let absr = past + r;
        for j in 0..kept {
            let c = off + j;
            let mut masked = c > absr;
            if let Some(w) = window {
                if absr + 1 > w && c + w <= absr {
                    masked = true;
                }
            }
            if masked {
                mask[r * kept + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask_t = Tensor::from_vec(mask, (n, kept), device).unwrap();
    let scores = qh
        .matmul(&kh.transpose(1, 2).unwrap().contiguous().unwrap())
        .unwrap()
        .broadcast_add(&mask_t)
        .unwrap();
    let probs = candle_nn::ops::softmax(&scores, candle_core::D::Minus1).unwrap();
    probs
        .matmul(&vh)
        .unwrap()
        .transpose(0, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

#[test]
#[ignore = "needs the CUDA card but no checkpoint; set NV_E4B_SERVING_TEST=1 -- the window-mask \
            transfer contract for the pfflash arm's flash half: candle-flash-attn windowed f16 \
            at the E4B sliding-layer geometry (8q/2kv heads, hd 256, chunk 512, sliding window \
            512, swnarrow kept slices, partial final chunk; full layers are hd 512 and do NOT \
            take flash) must agree with the eager cache_mask+softmax reference computed on the \
            SAME f16-rounded inputs, because agreement here isolates kernel/mask semantics \
            from input-rounding compounding, and an off-by-one window or misaligned diagonal \
            shows up as order-one wrong-key mass"]
fn e4b_cuda_pfflash_window_mask_transfers_cache_mask_semantics_on_f16_inputs() {
    require_gate("NV_E4B_SERVING_TEST");
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let (nh, nkv, hd) = (8usize, 2usize, 256usize);
    let sliding = 512usize;
    let cases: &[(usize, usize, Option<usize>, usize, f32, bool)] = &[
        (1536, 512, None, 2048, 1.0 / 16.0, false),
        (1536, 512, Some(sliding), 1024, 1.0 / 16.0, false),
        (1536, 264, Some(sliding), 776, 1.0 / 16.0, false),
        (0, 512, Some(sliding), 512, 1.0 / 16.0, false),
        (1536, 512, Some(sliding), 2048, 1.0 / 16.0, false),
        (0, 512, None, 512, 1.0 / 16.0, false),
        (0, 512, None, 512, 0.31, false),
        (1536, 512, None, 2048, 0.31, false),
        (512, 279, None, 791, 0.31, false),
        (0, 512, None, 512, 0.31, true),
        (512, 279, None, 791, 0.31, true),
        (1536, 512, None, 2048, 0.31, true),
    ];
    for &(past, n, window, kept, qk_scale, causal_via_local_window) in cases {
        let total = past + n;
        assert!(kept <= total);
        let q = Tensor::randn(0f32, qk_scale, (1, n, nh, hd), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let k = Tensor::randn(0f32, qk_scale, (1, kept, nkv, hd), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, kept, nkv, hd), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let cfg = nv_layers::attn::AttnConfig {
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            softmax_scale: 1.0,
            causal: true,
        };
        let window_left = match (window, causal_via_local_window) {
            (Some(w), _) => Some(w.saturating_sub(1)),
            (None, true) => Some(kept),
            (None, false) => None,
        };
        let got = nv_layers::attn::flash_attn_windowed(&q, &k, &v, &cfg, window_left, Some(0))
            .expect("flash_attn_windowed")
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = eager_reference_on_the_same_f16_rounded_inputs(
            &q, &k, &v, n, past, total, kept, window, nh, nkv, hd, &device,
        );
        assert_eq!(got.len(), want.len());
        let mut max_abs = 0f32;
        let mut max_i = 0usize;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let d = (g - w).abs();
            if d > max_abs {
                max_abs = d;
                max_i = i;
            }
        }
        eprintln!(
            "pfflash mask-transfer past={past} n={n} window={window:?} kept={kept} \
             qk_scale={qk_scale} via_local={causal_via_local_window} \
             max_abs={max_abs:.5} at flat={max_i} got={} want={}",
            got[max_i], want[max_i]
        );
        let tol = if qk_scale > 0.1 {
            10.0 * PFFLASH_MASK_TRANSFER_TOL_5E2_SEPARATES_FLASH_ACCUM_ROUNDING_FROM_ORDER_ONE_WRONG_KEY_MASS
        } else {
            PFFLASH_MASK_TRANSFER_TOL_5E2_SEPARATES_FLASH_ACCUM_ROUNDING_FROM_ORDER_ONE_WRONG_KEY_MASS
        };
        assert!(
            max_abs < tol,
            "flash window/mask semantics diverge from the eager cache_mask reference at \
             past={past} n={n} window={window:?} kept={kept} qk_scale={qk_scale} \
             via_local={causal_via_local_window}: max_abs={max_abs}"
        );
    }
}

const PFFLASH_AB_DEPTHS_8192_MATCHES_THE_SWNARROW_GATE_AND_1800_ENDS_ON_A_PARTIAL_264_CHUNK:
    [usize; 2] = [8192, 1800];
const PFFLASH_AB_LOGIT_ABS_FLOOR_1E4_WHEN_THE_REBUILD_CONTROL_IS_EXACT: f32 = 1e-4;
const PFFLASH_AB_CONTROL_MARGIN_2X_BOUNDS_THE_ARM_BY_THE_MEASURED_FRESH_PROCESS_REBUILD_NOISE_BECAUSE_E4B_W4A16_PREFILL_LOGITS_CARRY_ATOMIC_ACCUMULATION_NOISE_ON_REBUILD:
    f32 = 2.0;
const PFFLASH_AB_OFF_REBUILDS_3_BECAUSE_ONE_CONTROL_PAIR_UNDERSTATES_THE_REBUILD_NOISE_SPREAD:
    usize = 3;
const PFFLASH_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM: &str =
    "NV_E4B_PFFLASH_AB_CHILD_ROW_OUT";
const PFFLASH_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC: &str =
    "NV_E4B_PFFLASH_AB_CHILD_DEPTH";

#[test]
#[ignore = "re-execs the test binary so EACH arm loads the gemma-4 E4B serving checkpoint in a \
            FRESH process (same structure as the swnarrow gate); set NV_E4B_SERVING_TEST=1 -- \
            the flash-attn prefill identity contract: a chunk-512 prefill with \
            NV_E4B_PREFILL_FLASH_ATTN=1 (sliding layers hd256 via candle-flash-attn f16 \
            windowed, full layers hd512 via candle-flash-attn f16 causal -- the default since \
            candle 0.11 fixed the sm120 hdim512 forward) must keep the eager-path \
            prefill argmax \
            and stay within 2x the measured off-vs-off fresh-process rebuild noise, at depths \
            8192 (many whole-chunk seams) and 1800 (partial final chunk); both arms run on the \
            swnarrow serving basis so the only delta is the attention kernels and their \
            half-precision input rounding (E4B W4A16 prefill logits carry atomic accumulation noise across \
            rebuilds, so exact equality is not the property this path has; measured arm \
            shifts sit at 0.97-1.55x the rebuild-noise control across runs, so the strict \
            swnarrow pattern holds with no near-tie escape)"]
fn e4b_cuda_prefill_flash_attn_keeps_argmax_and_stays_within_rebuild_noise_of_the_eager_path() {
    require_gate("NV_E4B_SERVING_TEST");
    if let Ok(row_out) =
        std::env::var(PFFLASH_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM)
    {
        let depth: usize =
            std::env::var(PFFLASH_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC)
                .expect("child depth env")
                .parse()
                .expect("child depth parse");
        let prompt: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        ab_child_writes_last_prefill_row_le_f32(&row_out, &prompt);
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let (_dir, ckpt, _basis) = serving_checkpoint_dir();

    for depth in PFFLASH_AB_DEPTHS_8192_MATCHES_THE_SWNARROW_GATE_AND_1800_ENDS_ON_A_PARTIAL_264_CHUNK
    {
        let run_arm = |flash_on: bool, tag: &str| -> Vec<f32> {
            let exe = std::env::current_exe().expect("test binary path");
            let row_out = std::env::temp_dir().join(format!(
                "e4b-pfflash-ab-{}-{depth}-{tag}.f32row",
                std::process::id()
            ));
            let mut cmd = std::process::Command::new(exe);
            cmd.args([
                "--ignored",
                "--exact",
                "e4b_cuda_prefill_flash_attn_keeps_argmax_and_stays_within_rebuild_noise_of_the_eager_path",
                "--nocapture",
            ]);
            cmd.env(
                PFFLASH_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM,
                &row_out,
            );
            cmd.env(
                PFFLASH_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC,
                depth.to_string(),
            );
            let narrow_env = nv_models::gemma4_e4b::NV_E4B_PREFILL_SLIDING_NARROW_ENV_OPT_IN_1_SCORES_THE_35_SLIDING_LAYERS_AGAINST_ONLY_THE_LAST_WINDOW_512_PLUS_CHUNK_KV_ROWS_WITH_A_SHIFTED_MASK_BECAUSE_THE_SLIDING_MASK_ALREADY_DISCARDS_EVERY_OLDER_COLUMN_SO_FULL_HISTORY_SCORING_IS_PURE_WASTE_KV_SHARING_NEVER_CROSSES_LAYER_TYPES_AND_THE_CACHE_ITSELF_STAYS_FULL_HISTORY_FOR_DECODE;
            cmd.env(narrow_env, "1");
            let flash_env = nv_models::gemma4_e4b::NV_E4B_PREFILL_FLASH_ATTN_ENV_OPT_IN_1_SLIDING_LAYERS_HD256_TAKE_CANDLE_FLASH_ATTN_F16_NATIVE_GQA_WINDOW_LEFT_511_AND_FULL_LAYERS_HD512_TAKE_BF16_TENSOR_CORE_GEMMS_WITH_FUSED_SOFTMAX_BECAUSE_THE_HDIM512_FLASH_FWD_KERNEL_RETURNS_WRONG_MASS_ON_SM120_FOR_EVERY_DTYPE_AND_WINDOW_MODE_WHILE_DECODE_N1_KEEPS_ATTN_DECODE_F32_UNTOUCHED;
            if flash_on {
                cmd.env(flash_env, "1");
            } else {
                cmd.env_remove(flash_env);
            }
            let out = cmd.output().expect("spawn fresh-process arm");
            eprintln!("{}", String::from_utf8_lossy(&out.stderr));
            assert!(
                out.status.success(),
                "fresh-process arm {tag} at depth {depth} failed:\n{}",
                String::from_utf8_lossy(&out.stdout)
            );
            let bytes = std::fs::read(&row_out)
                .unwrap_or_else(|e| panic!("child row file {}: {e}", row_out.display()));
            std::fs::remove_file(&row_out).ok();
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        let rows_off: Vec<Vec<f32>> = (0
            ..PFFLASH_AB_OFF_REBUILDS_3_BECAUSE_ONE_CONTROL_PAIR_UNDERSTATES_THE_REBUILD_NOISE_SPREAD)
            .map(|i| run_arm(false, &format!("off{i}")))
            .collect();
        let row_on = run_arm(true, "on");
        assert_eq!(
            rows_off[0].len(),
            row_on.len(),
            "arms returned different vocab widths at depth {depth}"
        );
        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
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
        let mut control = 0f32;
        for i in 0..rows_off.len() {
            for j in i + 1..rows_off.len() {
                control = control.max(max_abs(&rows_off[i], &rows_off[j]));
            }
        }
        let arm = rows_off
            .iter()
            .map(|off| max_abs(off, &row_on))
            .fold(0f32, f32::max);
        eprintln!(
            "pfflash AB ckpt={ckpt} depth={depth} argmax_off={} argmax_on={} \
             control_logit_max_abs={control:.6} arm_logit_max_abs={arm:.6} \
             arm_over_control={:.2}",
            argmax(&rows_off[0]),
            argmax(&row_on),
            if control > 0.0 { arm / control } else { f32::INFINITY }
        );
        for off in &rows_off {
            assert_eq!(
                argmax(off),
                argmax(&rows_off[0]),
                "off rebuilds disagree on argmax at depth {depth}: the control itself is unstable"
            );
        }
        assert_eq!(
            argmax(&rows_off[0]),
            argmax(&row_on),
            "flash-attn prefill flipped the argmax at depth {depth}"
        );
        let ceiling = (control
            * PFFLASH_AB_CONTROL_MARGIN_2X_BOUNDS_THE_ARM_BY_THE_MEASURED_FRESH_PROCESS_REBUILD_NOISE_BECAUSE_E4B_W4A16_PREFILL_LOGITS_CARRY_ATOMIC_ACCUMULATION_NOISE_ON_REBUILD)
            .max(PFFLASH_AB_LOGIT_ABS_FLOOR_1E4_WHEN_THE_REBUILD_CONTROL_IS_EXACT);
        assert!(
            arm <= ceiling,
            "flash-attn prefill moved last-row logits by {arm} at depth {depth}, above the \
             rebuild-noise ceiling {ceiling} (off-vs-off control {control})"
        );
    }
}

const FULLFLASH_AB_DEPTHS_8192_MATCHES_THE_PFFLASH_GATE_AND_1800_ENDS_ON_A_PARTIAL_264_CHUNK_EXERCISING_THE_BOTTOM_RIGHT_CAUSAL_ALIGNMENT:
    [usize; 2] = [8192, 1800];
const FULLFLASH_AB_LOGIT_ABS_FLOOR_1E4_WHEN_THE_REBUILD_CONTROL_IS_EXACT: f32 = 1e-4;
const FULLFLASH_AB_CONTROL_MARGIN_2X_BOUNDS_THE_ARM_BY_THE_MEASURED_FRESH_PROCESS_REBUILD_NOISE_BECAUSE_E4B_W4A16_PREFILL_LOGITS_CARRY_ATOMIC_ACCUMULATION_NOISE_ON_REBUILD:
    f32 = 2.0;
const FULLFLASH_AB_OFF_REBUILDS_3_BECAUSE_ONE_CONTROL_PAIR_UNDERSTATES_THE_REBUILD_NOISE_SPREAD:
    usize = 3;
const FULLFLASH_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM: &str =
    "NV_E4B_FULLFLASH_AB_CHILD_ROW_OUT";
const FULLFLASH_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC: &str =
    "NV_E4B_FULLFLASH_AB_CHILD_DEPTH";

#[test]
#[ignore = "re-execs the test binary so EACH arm loads the gemma-4 E4B serving checkpoint in a \
            FRESH process (same structure as the pfflash gate); set NV_E4B_SERVING_TEST=1 -- \
            the hdim512-flash full-layer identity contract: under NV_E4B_PREFILL_FLASH_ATTN=1 \
            full layers hd512 default to candle-flash-attn f16 causal (candle 0.11's repaired \
            sm120 hdim512 forward); that default arm must keep the prefill argmax of the \
            NV_E4B_PREFILL_FULL_FLASH_HD512=0 bf16 tensor-core escape and stay within 2x the \
            measured escape-vs-escape fresh-process rebuild noise, at depths 8192 (many \
            whole-chunk seams) and 1800 (partial final chunk, exercising the bottom-right \
            causal alignment); both arms run on the swnarrow+pfflash serving basis so the only \
            delta is the full-layer attention route and its half-precision input rounding"]
fn e4b_cuda_prefill_full_flash_hd512_keeps_argmax_and_stays_within_rebuild_noise_of_the_bf16_route()
{
    require_gate("NV_E4B_SERVING_TEST");
    if let Ok(row_out) =
        std::env::var(FULLFLASH_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM)
    {
        let depth: usize =
            std::env::var(FULLFLASH_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC)
                .expect("child depth env")
                .parse()
                .expect("child depth parse");
        let prompt: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        ab_child_writes_last_prefill_row_le_f32(&row_out, &prompt);
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let (_dir, ckpt, _basis) = serving_checkpoint_dir();

    for depth in FULLFLASH_AB_DEPTHS_8192_MATCHES_THE_PFFLASH_GATE_AND_1800_ENDS_ON_A_PARTIAL_264_CHUNK_EXERCISING_THE_BOTTOM_RIGHT_CAUSAL_ALIGNMENT
    {
        let run_arm = |full_flash_on: bool, tag: &str| -> Vec<f32> {
            let exe = std::env::current_exe().expect("test binary path");
            let row_out = std::env::temp_dir().join(format!(
                "e4b-fullflash-ab-{}-{depth}-{tag}.f32row",
                std::process::id()
            ));
            let mut cmd = std::process::Command::new(exe);
            cmd.args([
                "--ignored",
                "--exact",
                "e4b_cuda_prefill_full_flash_hd512_keeps_argmax_and_stays_within_rebuild_noise_of_the_bf16_route",
                "--nocapture",
            ]);
            cmd.env(
                FULLFLASH_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM,
                &row_out,
            );
            cmd.env(
                FULLFLASH_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC,
                depth.to_string(),
            );
            let narrow_env = nv_models::gemma4_e4b::NV_E4B_PREFILL_SLIDING_NARROW_ENV_OPT_IN_1_SCORES_THE_35_SLIDING_LAYERS_AGAINST_ONLY_THE_LAST_WINDOW_512_PLUS_CHUNK_KV_ROWS_WITH_A_SHIFTED_MASK_BECAUSE_THE_SLIDING_MASK_ALREADY_DISCARDS_EVERY_OLDER_COLUMN_SO_FULL_HISTORY_SCORING_IS_PURE_WASTE_KV_SHARING_NEVER_CROSSES_LAYER_TYPES_AND_THE_CACHE_ITSELF_STAYS_FULL_HISTORY_FOR_DECODE;
            cmd.env(narrow_env, "1");
            let flash_env = nv_models::gemma4_e4b::NV_E4B_PREFILL_FLASH_ATTN_ENV_OPT_IN_1_SLIDING_LAYERS_HD256_TAKE_CANDLE_FLASH_ATTN_F16_NATIVE_GQA_WINDOW_LEFT_511_AND_FULL_LAYERS_HD512_TAKE_BF16_TENSOR_CORE_GEMMS_WITH_FUSED_SOFTMAX_BECAUSE_THE_HDIM512_FLASH_FWD_KERNEL_RETURNS_WRONG_MASS_ON_SM120_FOR_EVERY_DTYPE_AND_WINDOW_MODE_WHILE_DECODE_N1_KEEPS_ATTN_DECODE_F32_UNTOUCHED;
            cmd.env(flash_env, "1");
            let full_flash_env = nv_models::gemma4_e4b::NV_E4B_PREFILL_FULL_FLASH_HD512_ENV_OPT_OUT_0_UNDER_NV_E4B_PREFILL_FLASH_ATTN_FULL_ATTENTION_HD512_LAYERS_DEFAULT_TO_CANDLE_FLASH_ATTN_F16_NATIVE_GQA_CAUSAL_BECAUSE_CANDLE_011_FIXED_THE_SM120_HDIM512_FORWARD_AND_FLASH_WINS_1_09X_AT_8K_TO_1_39X_AT_120K_ON_THE_HD512_FLASHVSBF16_ROWS_WHILE_0_SELECTS_THE_BF16_TENSOR_CORE_GEMM_ROUTE;
            if full_flash_on {
                cmd.env_remove(full_flash_env);
            } else {
                cmd.env(full_flash_env, "0");
            }
            let out = cmd.output().expect("spawn fresh-process arm");
            eprintln!("{}", String::from_utf8_lossy(&out.stderr));
            assert!(
                out.status.success(),
                "fresh-process arm {tag} at depth {depth} failed:\n{}",
                String::from_utf8_lossy(&out.stdout)
            );
            let bytes = std::fs::read(&row_out)
                .unwrap_or_else(|e| panic!("child row file {}: {e}", row_out.display()));
            std::fs::remove_file(&row_out).ok();
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        let rows_off: Vec<Vec<f32>> = (0
            ..FULLFLASH_AB_OFF_REBUILDS_3_BECAUSE_ONE_CONTROL_PAIR_UNDERSTATES_THE_REBUILD_NOISE_SPREAD)
            .map(|i| run_arm(false, &format!("off{i}")))
            .collect();
        let row_on = run_arm(true, "on");
        assert_eq!(
            rows_off[0].len(),
            row_on.len(),
            "arms returned different vocab widths at depth {depth}"
        );
        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
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
        let mut control = 0f32;
        for i in 0..rows_off.len() {
            for j in i + 1..rows_off.len() {
                control = control.max(max_abs(&rows_off[i], &rows_off[j]));
            }
        }
        let arm = rows_off
            .iter()
            .map(|off| max_abs(off, &row_on))
            .fold(0f32, f32::max);
        eprintln!(
            "fullflash AB ckpt={ckpt} depth={depth} argmax_off={} argmax_on={} \
             control_logit_max_abs={control:.6} arm_logit_max_abs={arm:.6} \
             arm_over_control={:.2}",
            argmax(&rows_off[0]),
            argmax(&row_on),
            if control > 0.0 { arm / control } else { f32::INFINITY }
        );
        for off in &rows_off {
            assert_eq!(
                argmax(off),
                argmax(&rows_off[0]),
                "off rebuilds disagree on argmax at depth {depth}: the control itself is unstable"
            );
        }
        assert_eq!(
            argmax(&rows_off[0]),
            argmax(&row_on),
            "hd512 full-layer flash flipped the argmax at depth {depth}: the default flash \
             route is not identity-safe here, re-contain via NV_E4B_PREFILL_FULL_FLASH_HD512=0"
        );
        let ceiling = (control
            * FULLFLASH_AB_CONTROL_MARGIN_2X_BOUNDS_THE_ARM_BY_THE_MEASURED_FRESH_PROCESS_REBUILD_NOISE_BECAUSE_E4B_W4A16_PREFILL_LOGITS_CARRY_ATOMIC_ACCUMULATION_NOISE_ON_REBUILD)
            .max(FULLFLASH_AB_LOGIT_ABS_FLOOR_1E4_WHEN_THE_REBUILD_CONTROL_IS_EXACT);
        assert!(
            arm <= ceiling,
            "hd512 full-layer flash moved last-row logits by {arm} at depth {depth}, above the \
             rebuild-noise ceiling {ceiling} (off-vs-off control {control})"
        );
    }
}

const W4A8_AB_DEPTHS_8192_MATCHES_THE_SWNARROW_GATE_AND_1800_ENDS_ON_A_PARTIAL_264_CHUNK: [usize;
    2] = [8192, 1800];
const W4A8_AB_OFF_REBUILDS_3_BECAUSE_ONE_CONTROL_PAIR_UNDERSTATES_THE_REBUILD_NOISE_SPREAD: usize =
    3;
const W4A8_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM: &str =
    "NV_E4B_W4A8_AB_CHILD_ROW_OUT";
const W4A8_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC: &str =
    "NV_E4B_W4A8_AB_CHILD_DEPTH";

#[test]
#[ignore = "re-execs the test binary so EACH arm loads the gemma-4 E4B serving checkpoint in a \
            FRESH process (same structure as the swnarrow gate); set NV_E4B_SERVING_TEST=1 -- \
            the MLP W4A8-FP8 arm contract: a chunk-512 prefill with NV_E4B_MLP_W4A8FP8=1 (mlp \
            gate_up/down marlin GEMMs run per-row-fp8 activations; attn projections and the \
            decode gemv path stay a16) must keep the a16 prefill argmax at depths 8192 and \
            1800 EXCEPT across a near-tie the measured perturbation can close (off-arm top-2 \
            gap at the flip <= 2x the measured max-abs logit shift); the shift is printed \
            against the 3-rebuild a16 control -- fp8 activation quantization is lossy by \
            design, so a shift above rebuild noise is expected here and the numeric quality \
            bound for this arm is the chat-wrapped ppl gate; a confident-token flip fails"]
fn e4b_cuda_mlp_w4a8fp8_keeps_argmax_and_measures_logit_shift_vs_rebuild_noise() {
    require_gate("NV_E4B_SERVING_TEST");
    if let Ok(row_out) = std::env::var(W4A8_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM)
    {
        let depth: usize =
            std::env::var(W4A8_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC)
                .expect("child depth env")
                .parse()
                .expect("child depth parse");
        let prompt: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        ab_child_writes_last_prefill_row_le_f32(&row_out, &prompt);
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let (_dir, ckpt, _basis) = serving_checkpoint_dir();

    for depth in W4A8_AB_DEPTHS_8192_MATCHES_THE_SWNARROW_GATE_AND_1800_ENDS_ON_A_PARTIAL_264_CHUNK
    {
        let run_arm = |a8_on: bool, tag: &str| -> Vec<f32> {
            let exe = std::env::current_exe().expect("test binary path");
            let row_out = std::env::temp_dir().join(format!(
                "e4b-w4a8-ab-{}-{depth}-{tag}.f32row",
                std::process::id()
            ));
            let mut cmd = std::process::Command::new(exe);
            cmd.args([
                "--ignored",
                "--exact",
                "e4b_cuda_mlp_w4a8fp8_keeps_argmax_and_measures_logit_shift_vs_rebuild_noise",
                "--nocapture",
            ]);
            cmd.env(
                W4A8_AB_CHILD_ROW_OUT_ENV_MARKS_THE_FRESH_PROCESS_RE_EXEC_ARM,
                &row_out,
            );
            cmd.env(
                W4A8_AB_CHILD_DEPTH_ENV_CARRIES_THE_PARENTS_DEPTH_INTO_THE_RE_EXEC,
                depth.to_string(),
            );
            let a8_env = nv_models::gemma4_e4b::NV_E4B_MLP_W4A8FP8_ENV_OPT_IN_1_RUNS_MLP_MARLIN_GEMMS_WITH_PER_ROW_FP8_ACTIVATIONS_AND_X512_COMPENSATED_SCALES_BECAUSE_CHUNK_512_PREFILL_IS_THE_LARGE_M_REGIME_WHERE_FP8_TENSOR_CORES_PAY_WHILE_DECODE_M1_KEEPS_THE_GEMV_PATH_UNTOUCHED_AND_ATTN_PROJECTIONS_STAY_A16_PER_THE_QWEN_W4A8_MLP_ONLY_PRECEDENT;
            if a8_on {
                cmd.env(a8_env, "1");
            } else {
                cmd.env_remove(a8_env);
            }
            let out = cmd.output().expect("spawn fresh-process arm");
            eprintln!("{}", String::from_utf8_lossy(&out.stderr));
            assert!(
                out.status.success(),
                "fresh-process arm {tag} at depth {depth} failed:\n{}",
                String::from_utf8_lossy(&out.stdout)
            );
            let bytes = std::fs::read(&row_out)
                .unwrap_or_else(|e| panic!("child row file {}: {e}", row_out.display()));
            std::fs::remove_file(&row_out).ok();
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        let rows_off: Vec<Vec<f32>> = (0
            ..W4A8_AB_OFF_REBUILDS_3_BECAUSE_ONE_CONTROL_PAIR_UNDERSTATES_THE_REBUILD_NOISE_SPREAD)
            .map(|i| run_arm(false, &format!("off{i}")))
            .collect();
        let row_on = run_arm(true, "on");
        assert_eq!(
            rows_off[0].len(),
            row_on.len(),
            "arms returned different vocab widths at depth {depth}"
        );
        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
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
        let mut control = 0f32;
        for i in 0..rows_off.len() {
            for j in i + 1..rows_off.len() {
                control = control.max(max_abs(&rows_off[i], &rows_off[j]));
            }
        }
        let arm = rows_off
            .iter()
            .map(|off| max_abs(off, &row_on))
            .fold(0f32, f32::max);
        eprintln!(
            "w4a8fp8 AB ckpt={ckpt} depth={depth} argmax_off={} argmax_on={} \
             control_logit_max_abs={control:.6} arm_logit_max_abs={arm:.6} \
             arm_over_control={:.2}",
            argmax(&rows_off[0]),
            argmax(&row_on),
            if control > 0.0 { arm / control } else { f32::INFINITY }
        );
        for off in &rows_off {
            assert_eq!(
                argmax(off),
                argmax(&rows_off[0]),
                "off rebuilds disagree on argmax at depth {depth}: the control itself is unstable"
            );
        }
        let off_top = argmax(&rows_off[0]);
        let on_top = argmax(&row_on);
        if off_top != on_top {
            let gap_off = rows_off[0][off_top] - rows_off[0][on_top];
            eprintln!(
                "w4a8fp8 AB depth={depth} argmax flip {off_top}->{on_top} off_top2_gap_at_flip={gap_off:.4} \
                 arm_logit_max_abs={arm:.4}"
            );
            assert!(
                gap_off >= 0.0 && gap_off <= 2.0 * arm,
                "mlp w4a8-fp8 flipped the argmax at depth {depth} across an off-arm gap of \
                 {gap_off:.4}, which the measured per-logit perturbation {arm:.4} cannot close \
                 even with both logits moving toward each other -- that is a confident-token \
                 flip, not a near-tie, and the arm fails identity outright"
            );
        }
    }
}

const FIXED_CACHE_AB_DEPTH_8192_CROSSES_MANY_CHUNK_SEAMS_MATCHING_THE_BOOT_WARM_GATE: usize = 8192;
const FIXED_CACHE_AB_LOGIT_ABS_FLOOR_1E4_WHEN_THE_RERUN_CONTROL_IS_EXACT: f32 = 1e-4;
const FIXED_CACHE_AB_CONTROL_MARGIN_2X_BOUNDS_THE_ARM_BY_THE_MEASURED_IN_PROCESS_RERUN_NOISE_BECAUSE_E4B_W4A16_PREFILL_LOGITS_CARRY_ATOMIC_ACCUMULATION_NOISE:
    f32 = 2.0;

#[test]
#[ignore = "loads the gemma-4 E4B serving checkpoint; set NV_E4B_SERVING_TEST=1 -- the \
            fixed-capacity prefill KV cache identity contract at depth 8192: a chunk-512 \
            prefill through E4bPrefillFixedCache (NV_E4B_PREFILL_FIXED_CACHE arm) must keep \
            the cat-path prefill argmax and stay within 2x the measured in-process cat-vs-cat \
            rerun noise (E4B W4A16 prefill logits carry atomic accumulation noise, so exact \
            equality is not the property this path has)"]
fn e4b_cuda_prefill_fixed_cache_keeps_argmax_and_stays_within_rerun_noise_of_the_cat_path() {
    require_gate("NV_E4B_SERVING_TEST");
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let (dir, ckpt, _basis) = serving_checkpoint_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let (model, _build_s) = load_model(&dir, &device);
    let n_layers = model.config().num_hidden_layers;
    let depth = FIXED_CACHE_AB_DEPTH_8192_CROSSES_MANY_CHUNK_SEAMS_MATCHING_THE_BOOT_WARM_GATE;
    let prompt: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();

    let last_row = |logits: Tensor| -> Vec<f32> {
        logits
            .to_dtype(DType::F32)
            .expect("f32")
            .to_vec1()
            .expect("host row")
    };
    let run_cat = || -> Vec<f32> {
        let mut cache: Vec<Option<(Tensor, Tensor)>> = vec![None; n_layers];
        let mut off = 0usize;
        let mut last_logits = None;
        while off < prompt.len() {
            let c = (prompt.len() - off)
                .min(SERVING_PREFILL_CHUNK_512_MATCHES_RUN_SAMPLING_GEMMA4_E4B);
            last_logits = Some(
                model
                    .forward_step(&prompt[off..off + c], off, &mut cache)
                    .unwrap_or_else(|e| panic!("cat prefill chunk at {off}: {e:#}")),
            );
            off += c;
        }
        last_row(last_logits.expect("non-empty prompt"))
    };
    let run_fixed = || -> Vec<f32> {
        let mut fixed = model.prefill_fixed_cache(depth).expect("fixed cache");
        let mut off = 0usize;
        let mut last_logits = None;
        while off < prompt.len() {
            let c = (prompt.len() - off)
                .min(SERVING_PREFILL_CHUNK_512_MATCHES_RUN_SAMPLING_GEMMA4_E4B);
            last_logits = Some(
                model
                    .forward_step_fixed(&prompt[off..off + c], None, off, &mut fixed)
                    .unwrap_or_else(|e| panic!("fixed prefill chunk at {off}: {e:#}")),
            );
            off += c;
        }
        assert_eq!(fixed.seq_len(), depth, "fixed cache did not advance to depth");
        last_row(last_logits.expect("non-empty prompt"))
    };

    let row_cat = run_cat();
    let row_cat_control = run_cat();
    let row_fixed = run_fixed();
    assert_eq!(
        row_cat.len(),
        row_fixed.len(),
        "arms returned different vocab widths"
    );
    let max_abs = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
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
    let control = max_abs(&row_cat, &row_cat_control);
    let arm = max_abs(&row_cat, &row_fixed);
    eprintln!(
        "fixed-cache AB ckpt={ckpt} depth={depth} argmax_cat={} argmax_fixed={} \
         control_logit_max_abs={control:.6} arm_logit_max_abs={arm:.6}",
        argmax(&row_cat),
        argmax(&row_fixed)
    );
    assert_eq!(
        argmax(&row_cat),
        argmax(&row_fixed),
        "fixed-capacity prefill cache flipped the prefill argmax"
    );
    let ceiling = (control
        * FIXED_CACHE_AB_CONTROL_MARGIN_2X_BOUNDS_THE_ARM_BY_THE_MEASURED_IN_PROCESS_RERUN_NOISE_BECAUSE_E4B_W4A16_PREFILL_LOGITS_CARRY_ATOMIC_ACCUMULATION_NOISE)
        .max(FIXED_CACHE_AB_LOGIT_ABS_FLOOR_1E4_WHEN_THE_RERUN_CONTROL_IS_EXACT);
    assert!(
        arm <= ceiling,
        "fixed-capacity prefill cache moved last-row logits by {arm}, above the rerun-noise \
         ceiling {ceiling} (cat-vs-cat control {control})"
    );
}

#[test]
#[ignore = "loads the gemma-4 E4B serving checkpoint; set NV_E4B_SERVING_TEST=1 -- \
            serving-semantic chunk-512 eager prefill wall vs prompt depth (the exact loop \
            run_sampling_gemma4_e4b runs, fresh cat cache per run; \
            NV_E4B_PREFILL_FIXED_CACHE=1 swaps in the fixed-capacity in-place cache, \
            preallocation inside the clock as serving pays it), depths via NV_CTX_TOKENS \
            comma list (k suffix ok, default 8192,32768), warm basis: run 0 absorbs one-time \
            init, runs \
            1.. print E4B-CUDA-PREFILL machine lines; on the cat arm the last run at each \
            depth reuses its cache for a GraphedE4bDecoder steady-state decode (warm_step, \
            capture replay, then timed replay batches) and appends decode_* kv fields to \
            its line"]
fn e4b_cuda_chunked_prefill_tok_s_vs_prompt_depth_serving_last_row() {
    require_gate("NV_E4B_SERVING_TEST");
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let (dir, ckpt, basis) = serving_checkpoint_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let (model, _build_s) = load_model(&dir, &device);
    let n_layers = model.config().num_hidden_layers;
    let max_pos = model.config().max_position_embeddings;

    let depths = ctx_depths_from_env_one_grammar_for_every_test_in_this_binary_comma_list_each_entry_optional_k_suffix(
        &[8192, 32768],
    );
    const WARMUP_RUNS: usize = 1;
    const TIMED_RUNS: usize = 2;
    let fixed_arm = nv_models::gemma4_e4b::prefill_fixed_cache_env_opted_in();
    let narrow_arm = nv_models::gemma4_e4b::prefill_sliding_narrow_env_opted_in();
    let arm = match (fixed_arm, narrow_arm) {
        (true, true) => "fixed-swnarrow",
        (true, false) => "fixed",
        (false, true) => "cat-swnarrow",
        (false, false) => "cat",
    };
    let arm = if nv_models::gemma4_e4b::prefill_flash_attn_env_opted_in() {
        format!("{arm}-pfflash")
    } else {
        arm.to_string()
    };
    let arm = if nv_models::gemma4_e4b::prefill_flash_attn_env_opted_in()
        && nv_models::gemma4_e4b::prefill_full_flash_hd512_enabled()
    {
        format!("{arm}-fullflash")
    } else {
        arm
    };
    for depth in depths {
        assert!(depth + KV_HEADROOM_1024_MATCHES_THE_CTX_DEPTH_LADDER < max_pos);
        let prompt: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        for run in 0..WARMUP_RUNS + TIMED_RUNS {
            let mut cache: Vec<Option<(Tensor, Tensor)>> = vec![None; n_layers];
            device.synchronize().expect("sync before starting the clock");
            let t0 = Instant::now();
            let mut fixed = None;
            if fixed_arm {
                fixed = Some(model.prefill_fixed_cache(depth).expect("fixed cache"));
            }
            let mut off = 0usize;
            let mut last_logits = None;
            while off < prompt.len() {
                let c = (prompt.len() - off)
                    .min(SERVING_PREFILL_CHUNK_512_MATCHES_RUN_SAMPLING_GEMMA4_E4B);
                last_logits = Some(match fixed.as_mut() {
                    Some(f) => model
                        .forward_step_fixed(&prompt[off..off + c], None, off, f)
                        .unwrap_or_else(|e| panic!("fixed prefill chunk at {off}: {e:#}")),
                    None => model
                        .forward_step(&prompt[off..off + c], off, &mut cache)
                        .unwrap_or_else(|e| panic!("prefill chunk at {off}: {e:#}")),
                });
                off += c;
            }
            device.synchronize().expect("sync before stopping the clock");
            let dt = t0.elapsed().as_secs_f64();
            let seed = argmax_last_row(&last_logits.expect("non-empty prompt"));
            let tok_s = depth as f64 / dt;
            let mut decode_kv = String::new();
            if run + 1 == WARMUP_RUNS + TIMED_RUNS && !fixed_arm {
                let graph_batch = SERVING_GRAPH_BATCH_16_MATCHES_RUN_SAMPLING_GEMMA4_E4B;
                let decode_budget = (1 + TIMED_REPLAY_BATCHES_8_GIVE_128_TOKENS_FOR_A_STABLE_MEDIAN)
                    * graph_batch
                    + 1;
                let max_len = (depth + decode_budget + graph_batch + 8)
                    .next_power_of_two()
                    .max(64)
                    .min(max_pos);
                {
                    let mut dec = model
                        .graphed_decoder(&cache, depth, max_len)
                        .expect("graphed decoder");
                    let mut tok = dec.warm_step(seed).expect("warm step");
                    let batch = dec.replay_batch(tok, graph_batch).expect("capture replay");
                    tok = *batch.last().expect("non-empty batch");
                    device.synchronize().expect("sync after capture replay");
                    let mut batch_ms: Vec<f64> = Vec::new();
                    for _ in 0..TIMED_REPLAY_BATCHES_8_GIVE_128_TOKENS_FOR_A_STABLE_MEDIAN {
                        let td = Instant::now();
                        let batch = dec.replay_batch(tok, graph_batch).expect("timed replay");
                        tok = *batch.last().expect("non-empty batch");
                        device.synchronize().expect("sync after timed replay");
                        batch_ms.push(td.elapsed().as_secs_f64() * 1e3);
                    }
                    batch_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median_ms_tok = batch_ms[batch_ms.len() / 2] / graph_batch as f64;
                    let steps =
                        TIMED_REPLAY_BATCHES_8_GIVE_128_TOKENS_FOR_A_STABLE_MEDIAN * graph_batch;
                    decode_kv = format!(
                        " decode_median_ms_tok={median_ms_tok:.3} decode_tok_s={:.1} \
                         decode_steps={steps} decode_warmup_steps={} graph_batch={graph_batch} \
                         graph_max_len={max_len}",
                        1000.0 / median_ms_tok,
                        graph_batch + 1,
                    );
                }
                if let Err(e) = device.synchronize() {
                    eprintln!(
                        "drained the one-shot GraphedE4bDecoder teardown error after decode \
                         timing: {e:?}"
                    );
                }
            }
            if run >= WARMUP_RUNS {
                println!(
                    "E4B-CUDA-PREFILL checkpoint={ckpt} backend=cuda basis={basis} \
                     layers={n_layers} prompt_tokens={depth} prefill_s={dt:.2} \
                     tok_s={tok_s:.1} chunk={} run={run} seed={seed} arm={arm}{decode_kv}",
                    SERVING_PREFILL_CHUNK_512_MATCHES_RUN_SAMPLING_GEMMA4_E4B
                );
            } else {
                eprintln!(
                    "e4b-cuda-prefill warmup depth={depth} run={run} prefill_s={dt:.2} \
                     tok_s={tok_s:.1} seed={seed} arm={arm}"
                );
            }
        }
    }
}
