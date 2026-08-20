#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{LayerType, Qwen3Moe, Qwen3MoeKvCache, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use std::time::Instant;

mod ctx_timing_common;
mod hub_snapshot;
mod prime_ckpt_common;
mod common;
use common::qwen38_snapshot_dir_env_override_then_home_hub;

const PRIME_CKPT_FILL_MODE_REALCHUNK512_V1_FIXED_TOKEN_PATTERN: &str = "realchunk512v1";
const REAL_PRIME_CHUNK_512_MATCHES_THE_PROVEN_PREFILL_BLOCK: usize = 512;
const RESTORE_PARITY_DECODE_TOKENS_32_THE_TRACK_112_BIT_EXACTNESS_BAR: usize = 32;
const PRIME_DEPTH_DEFAULT_8K_THE_TRACK_112_TIMING_RUNG: usize = 8 * 1024;
const KV_SLOT_HEADROOM_16_BEYOND_THE_PARITY_STEPS: usize = 16;
const FIRST_DECODE_TOKEN_2000_FIXED_SO_FRESH_AND_RESTORED_START_IDENTICALLY: u32 = 2000;
const RESTORED_VS_FRESH_MS_TOK_BAND_2X_SAME_DEPTH_SAME_CACHE_GEOMETRY: f64 = 2.0;
const STEP0_LOGIT_MAX_ABS_DIFF_BAR_0_05_CORRUPTION_IS_O1_SUBSTRATE_DRIFT_IS_SMALLER: f32 = 0.05;

fn depth_from_env_first_entry_or_8k() -> usize {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => {
            let s = v.split(',').next().unwrap_or("").trim().to_string();
            let (num, mult) = match s.strip_suffix('k') {
                Some(n) => (n.to_string(), 1024usize),
                None => (s, 1usize),
            };
            num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
        }
        Err(_) => PRIME_DEPTH_DEFAULT_8K_THE_TRACK_112_TIMING_RUNG,
    }
}

fn ckpt_dir_env_or_home_tmp_wf_ckpt() -> PathBuf {
    prime_ckpt_common::prime_ckpt_dir_env_off_by_default_so_the_ladder_defaults_never_change()
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").expect("HOME");
            PathBuf::from(home).join("tmp/wf/ckpt")
        })
}

fn real_prefill_fixed_token_pattern(
    model: &Qwen3Moe,
    cache: &mut Qwen3MoeKvCache,
    depth: usize,
    device: &Device,
) {
    let mut pos = 0usize;
    while pos < depth {
        let n = REAL_PRIME_CHUNK_512_MATCHES_THE_PROVEN_PREFILL_BLOCK.min(depth - pos);
        let ids: Vec<u32> = (pos..pos + n).map(|i| 2000 + (i as u32 % 30000)).collect();
        let positions_v: Vec<i32> = (pos as i32..(pos + n) as i32).collect();
        let tokens = Tensor::from_vec(ids, (1usize, n), device).expect("tokens");
        let positions = Tensor::from_vec(positions_v, n, device).expect("positions");
        model
            .forward_with_cache_serving_prefill_last_row_logits_because_chat_prefill_samples_only_position_seq_minus_1(
                &tokens, &positions, cache,
            )
            .unwrap_or_else(|e| panic!("real prime chunk at pos {pos}: {e:#}"));
        pos += n;
    }
    device.synchronize().expect("sync after real prime");
    assert_eq!(
        cache.current_len(),
        depth,
        "real prime must commit exactly {depth} rows"
    );
}

fn argmax(row: &[f32]) -> u32 {
    let (mut bi, mut bv) = (0u32, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i as u32;
        }
    }
    bi
}

struct LadderRun {
    tokens: Vec<u32>,
    median_ms_tok: f64,
    step0_logits: Vec<f32>,
}

fn decode_argmax_ladder(
    model: &Qwen3Moe,
    cache: &mut Qwen3MoeKvCache,
    start_pos: usize,
    steps: usize,
    device: &Device,
) -> LadderRun {
    let mut cur = FIRST_DECODE_TOKEN_2000_FIXED_SO_FRESH_AND_RESTORED_START_IDENTICALLY;
    let mut out = Vec::with_capacity(steps);
    let mut step_ms = Vec::with_capacity(steps);
    let mut step0_logits = Vec::new();
    for s in 0..steps {
        let t0 = Instant::now();
        let tokens = Tensor::from_vec(vec![cur], (1usize, 1usize), device).expect("token");
        let positions =
            Tensor::from_vec(vec![(start_pos + s) as i32], 1usize, device).expect("position");
        let logits = model
            .forward_with_cache(&tokens, &positions, cache)
            .unwrap_or_else(|e| panic!("parity decode step {s}: {e:#}"));
        let row = logits
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("logits to host");
        step_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        if s == 0 {
            step0_logits = row.clone();
        }
        cur = argmax(&row);
        out.push(cur);
    }
    step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    LadderRun {
        tokens: out,
        median_ms_tok: step_ms[step_ms.len() / 2],
        step0_logits,
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 eager cuda dense arm; set NV_QWEN38_SERVING_TEST=1 -- the track-112 prime-checkpoint gate: REAL chunked prefill to depth (default 8k, first NV_CTX_TOKENS entry overrides), dump the primed cache (fp8 KV rows+scales, GDN LinAttnStates, current_len) under flock via tmp+rename (a pre-existing file under the same fingerprint is kept and restored, so a second invocation proves kill-resume across processes), then prove (a) a 32-token greedy decode ladder from the restored cache matches the fresh-prime ladder argmax-exact with step-0 logits inside the corruption bar, (b) restored decode ms/tok sits in the 2x band of fresh, and (c) restore is faster than re-priming; checkpoint dir is NV_KV_CKPT_DIR or ~/tmp/wf/ckpt, dump refuses if disk headroom is tight"]
fn qwen38_prime_ckpt_restore_matches_fresh_prime_argmax_32_tokens_and_beats_repriming() {
    if std::env::var("NV_QWEN38_SERVING_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_QWEN38_SERVING_TEST=1 to run this gate, or NV_MODELS_ALLOW_SKIP=1 to \
                 skip it on purpose; a prime-checkpoint gate that silently reports ok would \
                 let a corrupt restore path masquerade as a measured ladder"
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_QWEN38_SERVING_TEST=1 to run");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, &device)
        .expect("build Qwen3.8-27B on the eager cuda dense arm");
    drop(weights);

    let depth = depth_from_env_first_entry_or_8k();
    let slots = depth
        + RESTORE_PARITY_DECODE_TOKENS_32_THE_TRACK_112_BIT_EXACTNESS_BAR
        + KV_SLOT_HEADROOM_16_BEYOND_THE_PARITY_STEPS;
    let max_pos = model.config().max_position_embeddings;
    assert!(
        slots < max_pos,
        "depth {depth} + parity steps exceeds max_position_embeddings {max_pos}"
    );
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let ckpt_dir = ckpt_dir_env_or_home_tmp_wf_ckpt();
    let fingerprint =
        prime_ckpt_common::fingerprint_of_checkpoint_dims_depth_fillmode_and_cache_layout_version(
            &raw_cfg,
            model.config().num_hidden_layers,
            n_kv,
            hd,
            depth,
            PRIME_CKPT_FILL_MODE_REALCHUNK512_V1_FIXED_TOKEN_PATTERN,
        );

    let full_attn_slots = model
        .config()
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count();
    let expected_file_bytes =
        prime_ckpt_common::expected_ckpt_file_bytes_fp8_kv_rows_scales_plus_lin_state_slack(
            full_attn_slots,
            depth,
            n_kv,
            hd,
        );

    let mut cache_fresh = model.new_kv_cache(slots).expect("fresh kv cache");
    let prime_start = Instant::now();
    real_prefill_fixed_token_pattern(&model, &mut cache_fresh, depth, &device);
    let prime_s = prime_start.elapsed().as_secs_f64();

    let dump_start = Instant::now();
    let (ckpt_path, ckpt_provenance) = {
        let _lock =
            prime_ckpt_common::flock_exclusive_blocking_so_concurrent_lanes_serialize_per_fingerprint(
                &ckpt_dir,
                &fingerprint,
            )
            .unwrap_or_else(|e| panic!("flock: {e:#}"));
        let existing = prime_ckpt_common::ckpt_file_path(&ckpt_dir, &fingerprint);
        if existing.exists() {
            (existing, "prior_process_kill_resume")
        } else {
            let p = prime_ckpt_common::dump_cache_to_ckpt_file_tmp_then_rename_so_a_kill_never_leaves_a_torn_file(
                &cache_fresh,
                &ckpt_dir,
                &fingerprint,
                expected_file_bytes,
            )
            .unwrap_or_else(|e| panic!("dump primed cache: {e:#}"));
            (p, "this_process")
        }
    };
    let dump_s = dump_start.elapsed().as_secs_f64();
    let file_bytes = std::fs::metadata(&ckpt_path).expect("ckpt metadata").len();
    let file_mb = file_bytes as f64 / (1024.0 * 1024.0);
    assert!(
        file_bytes <= expected_file_bytes,
        "the on-disk checkpoint ({file_bytes} B) exceeds the size model \
         ({expected_file_bytes} B); an undercounting size model makes the disk-headroom \
         refusal meaningless"
    );

    let fresh = decode_argmax_ladder(
        &model,
        &mut cache_fresh,
        depth,
        RESTORE_PARITY_DECODE_TOKENS_32_THE_TRACK_112_BIT_EXACTNESS_BAR,
        &device,
    );
    drop(cache_fresh);

    let mut cache_restored = model.new_kv_cache(slots).expect("restored kv cache");
    let restore_start = Instant::now();
    {
        let _lock =
            prime_ckpt_common::flock_exclusive_blocking_so_concurrent_lanes_serialize_per_fingerprint(
                &ckpt_dir,
                &fingerprint,
            )
            .unwrap_or_else(|e| panic!("flock: {e:#}"));
        prime_ckpt_common::restore_cache_from_ckpt_file_checked(
            &mut cache_restored,
            &ckpt_path,
            &fingerprint,
        )
        .unwrap_or_else(|e| panic!("restore primed cache: {e:#}"));
    }
    device.synchronize().expect("sync after restore");
    let restore_s = restore_start.elapsed().as_secs_f64();
    assert_eq!(
        cache_restored.current_len(),
        depth,
        "restore must land current_len exactly at the primed depth"
    );
    let restored_lin_states = cache_restored
        .snapshot_lin_states()
        .expect("snapshot restored lin states");
    assert!(
        restored_lin_states.iter().any(|s| s.is_some()),
        "a real prime runs the GDN layers, so a restored cache with no LinAttnState \
         means the dump or restore dropped the linear-attention half of the state"
    );

    let restored = decode_argmax_ladder(
        &model,
        &mut cache_restored,
        depth,
        RESTORE_PARITY_DECODE_TOKENS_32_THE_TRACK_112_BIT_EXACTNESS_BAR,
        &device,
    );

    let wrong_fingerprint = format!("{fingerprint}-tampered");
    let refused = prime_ckpt_common::restore_cache_from_ckpt_file_checked(
        &mut cache_restored,
        &ckpt_path,
        &wrong_fingerprint,
    );
    let refused_msg = format!("{:#}", refused.expect_err("a fingerprint mismatch must refuse"));
    assert!(
        refused_msg.contains(nv_models::qwen3_5_moe::PRIME_CKPT_FINGERPRINT_MISMATCH),
        "mismatch refusal must carry the named error, got: {refused_msg}"
    );

    let step0_max_abs_diff = fresh
        .step0_logits
        .iter()
        .zip(&restored.step0_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "PRIME-CKPT qwen38 depth={depth} basis=eager_dense_real_prefill_chunk512_vs_flock_tmp_rename_ckpt prime_s={prime_s:.2} dump_s={dump_s:.2} restore_s={restore_s:.2} speedup={:.1}x file_mb={file_mb:.0} ckpt_provenance={ckpt_provenance} parity_tokens={} fresh==restored={} fresh_ms_tok={:.3} restored_ms_tok={:.3} step0_logit_max_abs_diff={step0_max_abs_diff:.6}",
        prime_s / restore_s,
        RESTORE_PARITY_DECODE_TOKENS_32_THE_TRACK_112_BIT_EXACTNESS_BAR,
        fresh.tokens == restored.tokens,
        fresh.median_ms_tok,
        restored.median_ms_tok
    );
    assert_eq!(
        fresh.tokens, restored.tokens,
        "the track-112 bit-exactness bar: a decode ladder from a restored prime must match \
         the fresh-prime ladder argmax-exact for all 32 tokens"
    );
    assert_eq!(
        fresh.step0_logits.len(),
        restored.step0_logits.len(),
        "step-0 logits rows must be the same vocab width"
    );
    assert!(
        step0_max_abs_diff <= STEP0_LOGIT_MAX_ABS_DIFF_BAR_0_05_CORRUPTION_IS_O1_SUBSTRATE_DRIFT_IS_SMALLER,
        "step-0 logits from the restored cache drift {step0_max_abs_diff} from the fresh \
         prime; a corrupt or misaligned restore shows O(1) drift, so this is a restore bug, \
         not substrate noise"
    );
    let ratio = restored.median_ms_tok / fresh.median_ms_tok;
    assert!(
        ratio <= RESTORED_VS_FRESH_MS_TOK_BAND_2X_SAME_DEPTH_SAME_CACHE_GEOMETRY
            && ratio >= 1.0 / RESTORED_VS_FRESH_MS_TOK_BAND_2X_SAME_DEPTH_SAME_CACHE_GEOMETRY,
        "decode ms/tok from the restored cache ({:.3}) is outside the 2x band of the fresh \
         prime ({:.3}); same depth and geometry must decode at the same speed, so the \
         restore landed a different effective cache",
        restored.median_ms_tok,
        fresh.median_ms_tok
    );
    assert!(
        restore_s < prime_s,
        "restore ({restore_s:.2}s) must beat re-priming ({prime_s:.2}s) at depth {depth}, \
         or the checkpoint is pure overhead"
    );
}
