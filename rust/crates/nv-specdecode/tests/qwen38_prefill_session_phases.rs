#![cfg(feature = "cuda")]

mod hub_dirs;

use candle_core::Device;
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_specdecode::qwen38_mtp::{
    mtp_chain_depth_from_env, mtp_draft_dir_override_from_env, Qwen38DenseMtpHead,
    Qwen38MtpGraphedDecodeSession, MTP_WEIGHTS_FILE_NAME,
};
use std::time::Instant;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const KV_HEADROOM_1024_MATCHES_THE_CTX_DEPTH_LADDER: usize = 1024;
const SESSION_ROUNDS_2_THE_FIRST_IS_THE_COLD_PROCESS_THE_SECOND_IS_A_WARM_SERVER_REQUEST: usize =
    2;

fn depth_from_env_default_8192() -> usize {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => {
            let s = v.trim();
            let (num, mult) = match s.strip_suffix(['k', 'K']) {
                Some(n) => (n, 1024usize),
                None => (s, 1usize),
            };
            num.parse::<usize>().expect("NV_CTX_TOKENS") * mult
        }
        Err(_) => 8192,
    }
}

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 dense arm plus the MTP head; set \
            NV_QWEN38_SERVING_TEST=1 -- walls each phase of the graphed mtp session start \
            (trunk chunked prefill via prefill_hidden, verify-lane capture, drafter kv \
            catch-up) at NV_CTX_TOKENS depth (plain integer, default 8192), twice in one \
            process: round 0 carries every one-time cold cost the per-depth ladder folds into \
            prefill_s, round 1 is what a warm server pays per request"]
fn qwen38_cuda_graphed_session_start_phase_walls_cold_then_warm() {
    require_gate("NV_QWEN38_SERVING_TEST");
    let dir = match std::env::var("NV_QWEN38_DIR") {
        Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
        _ => hub_dirs::snapshot(
            NVFP4_REPO,
            &["config.json", "tokenizer.json", MTP_WEIGHTS_FILE_NAME],
        )
        .expect("hydrated snapshot with the MTP shard, or set NV_QWEN38_DIR"),
    };
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = nv_weights::QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let n_layers = cfg.num_hidden_layers;
    let rev = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());
    let depth = depth_from_env_default_8192();
    let max_seq = depth + KV_HEADROOM_1024_MATCHES_THE_CTX_DEPTH_LADDER;
    assert!(max_seq < cfg.max_position_embeddings);
    let k = mtp_chain_depth_from_env();

    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("dense NVFP4 arm");
    drop(weights);
    let mut engine =
        GraphedQwen3Moe::new(base, &device, max_seq).expect("graphed engine");
    let mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &dir,
        engine.underlying(),
        &device,
    )
    .expect("MTP head");

    let prompt: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();

    for round in 0..SESSION_ROUNDS_2_THE_FIRST_IS_THE_COLD_PROCESS_THE_SECOND_IS_A_WARM_SERVER_REQUEST
    {
        let t_all = Instant::now();
        let t0 = Instant::now();
        engine.reset().expect("reset");
        engine.install_grouped_moe().expect("install");
        let mut mtp_kv = mtp.new_kv_cache(max_seq, &device).expect("drafter kv");
        device.synchronize().expect("sync");
        let setup_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let (last_row, hidden_t) = engine
            .prefill_hidden_serving_last_row_logits(&prompt)
            .expect("trunk chunked prefill");
        device.synchronize().expect("sync");
        let trunk_prefill_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        engine.ensure_verify_lane(k + 1).expect("verify lane");
        device.synchronize().expect("sync");
        let verify_capture_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let mut best = 0usize;
        let mut val = f32::NEG_INFINITY;
        for (i, &v) in last_row.iter().enumerate() {
            if v.is_finite() && v > val {
                val = v;
                best = i;
            }
        }
        let anchor = best as u32;
        let hidden_t = mtp
            .trunk_hidden_rows_as_drafter_anchor_input(&hidden_t)
            .expect("anchor input rows");
        let _base_hidden = hidden_t
            .narrow(1, prompt.len() - 1, 1)
            .expect("last row")
            .contiguous()
            .expect("contiguous");
        device.synchronize().expect("sync");
        let anchor_prep_s = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        mtp.prefill_prompt_kv_shifted_by_one_with_position_zero_on_a_zero_hidden(
            engine.underlying(),
            &prompt,
            &hidden_t,
            &mut mtp_kv,
        )
        .expect("drafter catch-up");
        device.synchronize().expect("sync");
        let drafter_catchup_s = t0.elapsed().as_secs_f64();
        let total_s = t_all.elapsed().as_secs_f64();

        assert_eq!(engine.current_pos(), depth);
        assert_eq!(mtp_kv.len(), depth);
        eprintln!(
            "PREFILL-PHASES depth={depth} round={round} anchor={anchor} k={k} \
             setup_s={setup_s:.3} trunk_prefill_s={trunk_prefill_s:.3} \
             verify_capture_s={verify_capture_s:.3} anchor_prep_s={anchor_prep_s:.3} \
             drafter_catchup_s={drafter_catchup_s:.3} total_s={total_s:.3} \
             trunk_tok_s={:.1} session_tok_s={:.1}",
            depth as f64 / trunk_prefill_s,
            depth as f64 / total_s
        );
        let chunk = nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
            .filter(|&c| depth > c)
            .unwrap_or(depth);
        let bwarm = std::env::var(
            nv_models::graph_engine::NV_Q38_BOOT_WARM_PREFILL_ENV_OPT_IN_1_RUNS_ONE_THROWAWAY_CHUNK_AT_ENGINE_BUILD_BECAUSE_THE_FIRST_REAL_CHUNK_PAYS_5S_OF_ONE_TIME_MODULE_AND_WORKSPACE_INIT_LANDING_IN_TTFT,
        )
        .ok()
        .as_deref()
            == Some("1");
        println!(
            "QWEN38-CUDA-PREFILL checkpoint=unsloth/Qwen3.8-27B-NVFP4@{rev} backend=cuda \
             layers={n_layers} prompt_tokens={depth} prefill_s={total_s:.2} tok_s={:.1} \
             chunk={chunk} run={round} basis=graphed_mtp_session_start_wall drafter=mtp k={k} \
             bwarm={} dwarm={} trunk_prefill_s={trunk_prefill_s:.2} \
             verify_capture_s={verify_capture_s:.2} drafter_catchup_s={drafter_catchup_s:.2}",
            depth as f64 / total_s,
            bwarm as u8,
            nv_specdecode::qwen38_mtp::drafter_boot_warm_selected_from_env() as u8,
        );
    }
}

const DWARM_AB_PROMPT_512_CROSSES_NO_CHUNK_SEAM_THE_WARM_TOUCHES_ONLY_PER_CALL_SHAPES: usize = 512;
const DWARM_AB_ROUNDS_6_EXERCISE_DRAFT_VERIFY_COMMIT_AND_THE_ROUND_CATCHUP: usize = 6;
const DWARM_AB_KV_HEADROOM_64_FITS_EVERY_ROUND: usize = 64;
const DWARM_AB_LOGIT_ABS_CEILING_1E4_THE_GRAPHED_QWEN_PATH_IS_BITWISE_SO_ANY_MORE_IS_LEAKED_STATE:
    f32 = 1e-4;
const DWARM_AB_PROBE_CATCHUP_TOKENS: [u32; 3] = [5, 7, 11];
const DWARM_AB_PROBE_DRAFT_TOKEN: u32 = 13;

fn dwarm_probe_hidden_rows_deterministic_lcg(
    rows: usize,
    hidden: usize,
    dtype: candle_core::DType,
    device: &Device,
) -> candle_core::Tensor {
    let vals: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i * 37) % 101) as f32 / 101.0 - 0.5)
        .collect();
    candle_core::Tensor::from_vec(vals, (1usize, rows, hidden), device)
        .expect("probe hidden rows")
        .to_dtype(dtype)
        .expect("probe hidden dtype")
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 dense arm once plus the MTP head three times; \
            set NV_QWEN38_SERVING_TEST=1 -- the drafter boot-warm hygiene contract: a head built \
            with NV_Q38_BOOT_WARM_DRAFTER=1 (one throwaway catch-up and draft into a scratch kv \
            dropped before return) must emit the same graphed-session tokens and the same \
            drafter logits on fixed inputs as a head built without it, and a second warm-off \
            head built AFTER the warmed one must match the first, proving the warm also left no \
            state in the shared trunk"]
fn qwen38_cuda_drafter_boot_warm_leaves_no_state_a_request_can_see() {
    require_gate("NV_QWEN38_SERVING_TEST");
    let dir = match std::env::var("NV_QWEN38_DIR") {
        Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
        _ => hub_dirs::snapshot(
            NVFP4_REPO,
            &["config.json", "tokenizer.json", MTP_WEIGHTS_FILE_NAME],
        )
        .expect("hydrated snapshot with the MTP shard, or set NV_QWEN38_DIR"),
    };
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = nv_weights::QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let depth = DWARM_AB_PROMPT_512_CROSSES_NO_CHUNK_SEAM_THE_WARM_TOUCHES_ONLY_PER_CALL_SHAPES;
    let max_seq = depth + DWARM_AB_KV_HEADROOM_64_FITS_EVERY_ROUND;
    let k = mtp_chain_depth_from_env();
    let prompt: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();

    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("dense NVFP4 arm");
    drop(weights);
    let mut engine = GraphedQwen3Moe::new(base, &device, max_seq).expect("graphed engine");

    let dwarm_env = nv_specdecode::qwen38_mtp::NV_Q38_BOOT_WARM_DRAFTER_ENV_OPT_IN_1_RUNS_ONE_THROWAWAY_CATCHUP_AND_DRAFT_AT_HEAD_BUILD_SIBLING_OF_THE_TRUNK_BOOT_WARM_BECAUSE_THE_DRAFTERS_FIRST_FORWARD_PAYS_ITS_OWN_ONE_TIME_INIT_THE_TRUNK_WARM_CHUNK_NEVER_TOUCHES;
    let dtype = engine.underlying().dtype();
    let hidden = engine.underlying().config().hidden_size;

    let mut run_arm = |warm_on: bool| -> (Vec<u32>, Vec<f32>) {
        if warm_on {
            std::env::set_var(dwarm_env, "1");
        } else {
            std::env::remove_var(dwarm_env);
        }
        let head = Qwen38DenseMtpHead::from_checkpoint(
            mtp_draft_dir_override_from_env().as_deref(),
            &dir,
            engine.underlying(),
            &device,
        )
        .expect("MTP head");
        let mut emitted: Vec<u32> = Vec::new();
        {
            let mut sess = Qwen38MtpGraphedDecodeSession::start(&mut engine, &head, k, &prompt)
                .expect("graphed mtp session");
            emitted.push(sess.anchor());
            for _ in 0..DWARM_AB_ROUNDS_6_EXERCISE_DRAFT_VERIFY_COMMIT_AND_THE_ROUND_CATCHUP {
                emitted.extend(sess.round().expect("round"));
            }
        }
        let probe_rows = dwarm_probe_hidden_rows_deterministic_lcg(
            DWARM_AB_PROBE_CATCHUP_TOKENS.len(),
            hidden,
            dtype,
            &device,
        );
        let mut probe_kv = head
            .new_kv_cache(DWARM_AB_PROBE_CATCHUP_TOKENS.len() + 2, &device)
            .expect("probe kv");
        head.catch_up_kv_recomputing_and_discarding_q_which_v1_prices_over_a_kv_only_projection(
            engine.underlying(),
            &DWARM_AB_PROBE_CATCHUP_TOKENS,
            &probe_rows,
            &mut probe_kv,
        )
        .expect("probe catch-up");
        let anchor_row = probe_rows
            .narrow(1, DWARM_AB_PROBE_CATCHUP_TOKENS.len() - 1, 1)
            .expect("anchor row")
            .contiguous()
            .expect("contiguous");
        let (logits, _dhidden) = head
            .forward_draft(
                engine.underlying(),
                &anchor_row,
                DWARM_AB_PROBE_DRAFT_TOKEN,
                &mut probe_kv,
            )
            .expect("probe draft");
        let row: Vec<f32> = logits
            .to_dtype(candle_core::DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("host row");
        (emitted, row)
    };

    let (emitted_off, logits_off) = run_arm(false);
    let (emitted_on, logits_on) = run_arm(true);
    let (emitted_off_control, logits_off_control) = run_arm(false);
    std::env::remove_var(dwarm_env);

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
            assert!(v.is_finite(), "non-finite drafter logit at {i}");
            if v > val {
                val = v;
                best = i;
            }
        }
        best
    };
    let on_abs = max_abs(&logits_off, &logits_on);
    let control_abs = max_abs(&logits_off, &logits_off_control);
    eprintln!(
        "drafter-warm AB depth={depth} k={k} rounds={} emitted_len={} \
         logit_max_abs_off_vs_on={on_abs:.6} logit_max_abs_off_vs_off={control_abs:.6} \
         argmax_off={} argmax_on={}",
        DWARM_AB_ROUNDS_6_EXERCISE_DRAFT_VERIFY_COMMIT_AND_THE_ROUND_CATCHUP,
        emitted_off.len(),
        argmax(&logits_off),
        argmax(&logits_on),
    );
    assert_eq!(
        emitted_off, emitted_off_control,
        "the warm-off rerun control diverged from itself; the gate cannot attribute any \
         off-vs-on difference to the drafter warm on this path"
    );
    assert_eq!(
        emitted_off, emitted_on,
        "drafter boot warm changed graphed-session emitted tokens; the throwaway catch-up left \
         state a request can see"
    );
    assert_eq!(argmax(&logits_off), argmax(&logits_on), "drafter boot warm flipped a draft argmax");
    assert!(
        on_abs <= DWARM_AB_LOGIT_ABS_CEILING_1E4_THE_GRAPHED_QWEN_PATH_IS_BITWISE_SO_ANY_MORE_IS_LEAKED_STATE,
        "drafter boot warm moved drafter logits by {on_abs} (ceiling {}, off-vs-off control {control_abs})",
        DWARM_AB_LOGIT_ABS_CEILING_1E4_THE_GRAPHED_QWEN_PATH_IS_BITWISE_SO_ANY_MORE_IS_LEAKED_STATE
    );
}
