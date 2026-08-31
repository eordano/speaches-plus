mod hub_dirs;

use std::time::Instant;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const CHECKPOINT_LABEL: &str = "unsloth/Qwen3.8-27B-NVFP4";

const TIMED_EMITTED_TOKENS_256_THE_SCOREBOARD_GENERATION_LENGTH: usize = 256;
const TIMED_EMITTED_TOKENS_ENV_OPTS_IN_A_LONGER_WINDOW_UNSET_KEEPS_THE_SCOREBOARD_256: &str =
    "NV_TIMED_EMITTED_TOKENS";

fn timed_emitted_tokens_from_env_default_the_scoreboard_256() -> usize {
    match std::env::var(TIMED_EMITTED_TOKENS_ENV_OPTS_IN_A_LONGER_WINDOW_UNSET_KEEPS_THE_SCOREBOARD_256)
    {
        Ok(v) => v
            .parse::<usize>()
            .expect("NV_TIMED_EMITTED_TOKENS must be a plain token count"),
        Err(_) => TIMED_EMITTED_TOKENS_256_THE_SCOREBOARD_GENERATION_LENGTH,
    }
}
const PLATEAU_WINDOW_4_STEPS_MEDIAN_VS_THE_PREVIOUS_4: usize = 4;
const PLATEAU_FLOOR_8_STEPS_THE_OLD_FIXED_WARMUP_THAT_DID_NOT_RAMP_CLOCKS: usize = 8;
const PLATEAU_RATIO_LAST_WINDOW_MEDIAN_WITHIN_10_PCT_OF_PREVIOUS: f64 = 0.10;
const PLATEAU_CAP_15_SECS_SO_A_SLOW_DEEP_RUNG_CANNOT_WARM_FOREVER: f64 = 15.0;
const PLAIN_ARM_WARMUP_CAP_200_STEPS_MATCHES_THE_CTX_SCALING_LADDERS: usize = 200;
const MTP_ARM_WARMUP_CAP_40_ROUNDS_BOUNDS_DEPTH_DRIFT_OF_THE_TIMED_WINDOW: usize = 40;
const KV_SLOT_HEADROOM_1024_COVERS_ANCHOR_WARMUP_TIMED_OVERSHOOT_AND_THE_512_ROW_VERIFY_GUARD:
    usize = 1024;

const WGPU_MTP_ARM_ENV_NV_Q3D_MTP_MATCHES_THE_CHAT_ENGINE_OPT_IN: &str = "NV_Q3D_MTP";

const TILED_PROMPT_PARAGRAPH_REPETITION_LANDS_IN_THE_HIGH_ACCEPT_DOC_REGIME_SO_ACCEPTANCE_HERE_IS_AN_UPPER_BRACKET_NOT_A_CHAT_NUMBER:
    &str = "Polynesian wayfinders crossed thousands of miles of open ocean by reading star \
     paths, swell patterns, and the flight of birds, long before the magnetic compass reached \
     Europe. Their voyaging canoes carried taro, breadfruit, and chickens between island chains \
     separated by weeks of sailing. ";

fn ctx_tokens_from_env_default_256_8k_196k() -> Vec<usize> {
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
        Err(_) => vec![256, 8 * 1024, 192 * 1024],
    }
}

fn median_of(ms: &[f64]) -> f64 {
    let mut sorted = ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn warmup_until_step_wall_plateaus(
    mut one_step_returning_tokens_emitted: impl FnMut() -> usize,
    step_cap: usize,
) -> (usize, usize) {
    let started = Instant::now();
    let mut ms: Vec<f64> = Vec::new();
    let mut tokens = 0usize;
    loop {
        let t0 = Instant::now();
        tokens += one_step_returning_tokens_emitted();
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
        let n = ms.len();
        let w = PLATEAU_WINDOW_4_STEPS_MEDIAN_VS_THE_PREVIOUS_4;
        if n >= PLATEAU_FLOOR_8_STEPS_THE_OLD_FIXED_WARMUP_THAT_DID_NOT_RAMP_CLOCKS && n >= 2 * w {
            let last = median_of(&ms[n - w..]);
            let prev = median_of(&ms[n - 2 * w..n - w]);
            if prev > 0.0
                && (last - prev).abs()
                    <= PLATEAU_RATIO_LAST_WINDOW_MEDIAN_WITHIN_10_PCT_OF_PREVIOUS * prev
            {
                return (n, tokens);
            }
        }
        if n >= step_cap
            || started.elapsed().as_secs_f64()
                >= PLATEAU_CAP_15_SECS_SO_A_SLOW_DEEP_RUNG_CANNOT_WARM_FOREVER
        {
            return (n, tokens);
        }
    }
}

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
    }
}

fn snapshot_rev_label(dir: &std::path::Path) -> String {
    dir.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into())
}

struct RoundTally {
    rounds: usize,
    drafted: usize,
    accepted: usize,
}

fn hist_window_delta_counts_only_grow(
    before: &std::collections::BTreeMap<usize, usize>,
    after: &std::collections::BTreeMap<usize, usize>,
) -> std::collections::BTreeMap<usize, usize> {
    let mut out = std::collections::BTreeMap::new();
    for (&len, &n) in after {
        let n0 = before.get(&len).copied().unwrap_or(0);
        assert!(
            n >= n0,
            "session histogram count for len={len} shrank {n0}->{n}: hist entries only grow, so \
             a shrink means the windowing snapshot came from a different session"
        );
        if n > n0 {
            out.insert(len, n - n0);
        }
    }
    out
}

fn len_colon_count_pairs(hist: &std::collections::BTreeMap<usize, usize>) -> String {
    if hist.is_empty() {
        return "empty".into();
    }
    hist.iter()
        .map(|(len, n)| format!("{len}:{n}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn adaptive_shape_suffix_appended_after_checkpoint_so_parse_and_append_kv_parsing_stays_additive(
    p_min_label: &str,
    timed_draft_len_hist: &std::collections::BTreeMap<usize, usize>,
    timed_accept_len_hist: &std::collections::BTreeMap<usize, usize>,
) -> String {
    format!(
        " p_min={p_min_label} draft_len_hist={} accept_len_hist={}",
        len_colon_count_pairs(timed_draft_len_hist),
        len_colon_count_pairs(timed_accept_len_hist),
    )
}

fn p_min_label_matching_the_k_sweep_flag() -> String {
    nv_specdecode::qwen38_mtp::mtp_draft_p_min_from_env()
        .map(|p| format!("{p:.2}"))
        .unwrap_or_else(|| "off".into())
}

#[test]
fn ctx_scaling_adaptive_shape_suffix_is_kv_pairs_the_runs_jsonl_parser_ignores_additively() {
    let mut before = std::collections::BTreeMap::new();
    before.insert(3usize, 5usize);
    let mut after = before.clone();
    *after.entry(3).or_insert(0) += 40;
    *after.entry(2).or_insert(0) += 7;
    let delta = hist_window_delta_counts_only_grow(&before, &after);
    assert_eq!(len_colon_count_pairs(&delta), "2:7,3:40");
    let mut accepts = std::collections::BTreeMap::new();
    accepts.insert(0usize, 3usize);
    accepts.insert(2usize, 12usize);
    let suffix =
        adaptive_shape_suffix_appended_after_checkpoint_so_parse_and_append_kv_parsing_stays_additive(
            "0.75", &delta, &accepts,
        );
    assert_eq!(
        suffix,
        " p_min=0.75 draft_len_hist=2:7,3:40 accept_len_hist=0:3,2:12"
    );
    assert_eq!(
        len_colon_count_pairs(&std::collections::BTreeMap::new()),
        "empty"
    );
}

#[allow(clippy::too_many_arguments)]
fn assert_mtp_window_not_degenerate_then_print_ctx_scaling_line(
    tag: &str,
    arm: &str,
    depth: usize,
    k: usize,
    max_round_overshoot: usize,
    tally: &RoundTally,
    timed_emitted: usize,
    warm_emitted: usize,
    wall_s: f64,
    round_ms: &[f64],
    fill_or_prefill: &str,
    rev: &str,
    adaptive_shape_suffix: &str,
) {
    assert!(
        tally.rounds > 0 && tally.drafted > 0,
        "degenerate mtp run at depth {depth}: {} rounds drafted {} tokens; a measure line from \
         a run that never drafted would enter runs.jsonl as a speculative number measured on \
         nothing",
        tally.rounds,
        tally.drafted
    );
    assert!(
        tally.accepted > 0,
        "zero acceptance over {} drafts at depth {depth}: the drafter never landed a token, so \
         effective tok/s here is plain decode wearing an mtp label; refuse the row and triage \
         the head (or the synthetic-state regime) instead",
        tally.drafted
    );
    let requested = timed_emitted_tokens_from_env_default_the_scoreboard_256();
    assert!(
        timed_emitted >= requested && timed_emitted <= requested + max_round_overshoot,
        "timed window emitted {timed_emitted} tokens, requested {requested} (+ at most \
         {max_round_overshoot} overshoot from the final round)"
    );
    assert!(
        wall_s.is_finite() && wall_s > 0.0,
        "non-positive wall {wall_s}s over the timed mtp window at depth {depth}"
    );
    let eff_tok_s = timed_emitted as f64 / wall_s;
    let eff_ms_tok = 1000.0 / eff_tok_s;
    let accept_rate = tally.accepted as f64 / tally.drafted as f64;
    eprintln!(
        "CTX-SCALING {tag} depth={depth} arm={arm} k={k} rounds={} steps={timed_emitted} \
         warmup_steps={warm_emitted} drafted={} accepted={} accept_rate={accept_rate:.3} \
         tokens_per_round={:.2} median_ms_tok={eff_ms_tok:.3} tok_s={eff_tok_s:.1} \
         round_ms_median={:.1} {fill_or_prefill} \
         checkpoint={CHECKPOINT_LABEL}@{rev}{adaptive_shape_suffix}",
        tally.rounds,
        tally.drafted,
        tally.accepted,
        timed_emitted as f64 / tally.rounds as f64,
        median_of(round_ms),
    );
}

fn print_plain_ctx_scaling_line(
    tag: &str,
    depth: usize,
    steps: usize,
    warm_steps: usize,
    step_ms: &[f64],
    fill_or_prefill: &str,
    rev: &str,
) {
    assert_eq!(
        steps,
        timed_emitted_tokens_from_env_default_the_scoreboard_256(),
        "plain control arm at depth {depth} timed {steps} steps instead of the requested count"
    );
    let median = median_of(step_ms);
    assert!(
        median.is_finite() && median > 0.0,
        "median decode step time is not a positive number: {median}"
    );
    eprintln!(
        "CTX-SCALING {tag} depth={depth} arm=plain steps={steps} warmup_steps={warm_steps} \
         median_ms_tok={median:.3} tok_s={:.1} {fill_or_prefill} \
         checkpoint={CHECKPOINT_LABEL}@{rev}",
        1000.0 / median,
    );
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B dense hybrid on wgpu; set NV_QWEN38_WGPU_TEST=1 -- \
            speculative (MTP-drafter) EFFECTIVE decode tok/s (emitted tokens / wall) vs KV depth \
            over the same synthetic state fill the plain ctx-scaling ladder uses; NV_Q3D_MTP=1 \
            selects the mtp arm (drafter KV buffers stay boot-zero under synthfill; the drafter \
            graph records a single stage1 arm, its replay loops never filter, per \
            DEPTH_GATE_SELECTS_AT_ENCODE in qwen3_5_dense_wgpu.rs), unset it for the plain \
            control arm whose median must reconcile with the established \
            qwen38_wgpu_decode_ms_per_token rows; run ONE depth per process via NV_CTX_TOKENS"]
fn qwen38_wgpu_mtp_effective_decode_tok_s_vs_context_depth_synthetic_state_fill_arm_keyed_on_nv_q3d_mtp(
) {
    use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseWgpu;
    use nv_specdecode::qwen38_mtp::{
        mtp_chain_depth_from_env, mtp_verify_replay_selected_from_env, run_mtp_verify_round,
    };

    require_gate("NV_QWEN38_WGPU_TEST");
    let mtp_arm = std::env::var(WGPU_MTP_ARM_ENV_NV_Q3D_MTP_MATCHES_THE_CHAT_ENGINE_OPT_IN)
        .ok()
        .as_deref()
        == Some("1");
    let dir = match std::env::var("NV_QWEN38_DIR") {
        Ok(d) if !d.is_empty() => {
            let p = std::path::PathBuf::from(&d);
            assert!(p.is_dir(), "NV_QWEN38_DIR={d} is not a directory");
            p
        }
        _ => hub_dirs::snapshot(NVFP4_REPO, &["config.json", "model_mtp.safetensors"])
            .unwrap_or_else(|| {
                panic!(
                    "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no hydrated {NVFP4_REPO} \
                     snapshot carrying the MTP shard; set NV_QWEN38_DIR"
                )
            }),
    };
    let rev = snapshot_rev_label(&dir);
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());

    let depths = ctx_tokens_from_env_default_256_8k_196k();
    let max_depth = depths.iter().copied().max().unwrap();
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseConfig::from_hf_json_str(&raw)
        .expect("qwen3.8 dense config");
    let max_seq =
        max_depth + KV_SLOT_HEADROOM_1024_COVERS_ANCHOR_WARMUP_TIMED_OVERSHOOT_AND_THE_512_ROW_VERIFY_GUARD;
    assert!(
        max_seq < cfg.max_position_embeddings,
        "a model declaring max_pos {} must serve depth {max_depth}, so a trip here is a config bug",
        cfg.max_position_embeddings
    );
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut gpu = Qwen3_5DenseWgpu::from_loader(cfg, &loader, max_seq)
        .expect("build Qwen3.8-27B on the qwen3_5 dense wgpu decoder");
    if mtp_arm {
        gpu.mtp_attach(&loader)
            .expect("attach the mtp.* drafter head (the mtp arm cannot run without it)");
    }
    drop(loader);
    let k = mtp_chain_depth_from_env();
    let replay = mtp_verify_replay_selected_from_env();

    let mtp_round = |gpu: &mut Qwen3_5DenseWgpu,
                         last: &mut u32,
                         tally: &mut RoundTally|
     -> usize {
        let rows = gpu.verify_max_rows();
        assert!(rows >= 2, "NV_WGPU_PREFILL_M did not build the M-row verify graph");
        assert!(
            gpu.current_pos() + rows.max(gpu.prefill_chunk_len()) <= max_seq,
            "kv headroom exhausted mid-window at pos {}: the serving loop would fall back to \
             plain decode here and the arm label would lie",
            gpu.current_pos()
        );
        let want = k.min(rows - 1);
        let drafts = gpu.mtp_draft_round(*last, want).expect("mtp draft round");
        let r = run_mtp_verify_round(gpu, *last, &drafts, replay).expect("mtp verify round");
        gpu.mtp_post_verify(&r.batch[1..r.accept.commit_len])
            .expect("mtp post verify");
        tally.rounds += 1;
        tally.drafted += drafts.len();
        tally.accepted += r.accept.draft_accepted;
        *last = *r.emitted.last().expect("a round always emits the bonus token");
        r.emitted.len()
    };

    for &depth in &depths {
        gpu.reset().expect("reset between depth arms");
        let fill_start = Instant::now();
        gpu.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(depth)
            .unwrap_or_else(|e| panic!("synthetic state fill at depth {depth}: {e:#}"));
        ctx.poll_blocking()
            .expect("drain synthetic fill writes before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(gpu.current_pos(), depth, "synthetic fill must land at the requested depth");
        if mtp_arm {
            assert!(gpu.mtp_active(), "mtp arm without an attached head");
            assert_eq!(
                gpu.mtp_len(),
                depth,
                "synthetic fill must leave the drafter KV length at the trunk depth"
            );
        }
        let mut last = gpu.decode_step(2000).expect("anchor decode step");
        let fill_label = format!("fill_s={fill_s:.1}");

        if mtp_arm {
            let mut tally = RoundTally { rounds: 0, drafted: 0, accepted: 0 };
            let (_warm_rounds, warm_emitted) = warmup_until_step_wall_plateaus(
                || mtp_round(&mut gpu, &mut last, &mut tally),
                MTP_ARM_WARMUP_CAP_40_ROUNDS_BOUNDS_DEPTH_DRIFT_OF_THE_TIMED_WINDOW,
            );
            let mut timed = RoundTally { rounds: 0, drafted: 0, accepted: 0 };
            let mut round_ms: Vec<f64> = Vec::new();
            let mut emitted = 0usize;
            let t0 = Instant::now();
            while emitted < timed_emitted_tokens_from_env_default_the_scoreboard_256() {
                let r0 = Instant::now();
                emitted += mtp_round(&mut gpu, &mut last, &mut timed);
                round_ms.push(r0.elapsed().as_secs_f64() * 1e3);
            }
            let wall_s = t0.elapsed().as_secs_f64();
            assert_eq!(
                gpu.current_pos(),
                depth + 1 + warm_emitted + emitted,
                "trunk pos must advance by exactly the anchor plus every emitted token"
            );
            assert_eq!(
                gpu.mtp_len(),
                gpu.current_pos(),
                "round catch-up must leave the drafter KV at the trunk committed length"
            );
            assert_mtp_window_not_degenerate_then_print_ctx_scaling_line(
                "qwen38-wgpu-mtp-depth",
                "mtp",
                depth,
                k,
                k,
                &timed,
                emitted,
                warm_emitted,
                wall_s,
                &round_ms,
                &fill_label,
                &rev,
                "",
            );
        } else {
            let (warm_steps, _warm_tokens) = warmup_until_step_wall_plateaus(
                || {
                    last = gpu.decode_step(last).expect("warm decode step");
                    1
                },
                PLAIN_ARM_WARMUP_CAP_200_STEPS_MATCHES_THE_CTX_SCALING_LADDERS,
            );
            let mut step_ms: Vec<f64> = Vec::new();
            for _ in 0..timed_emitted_tokens_from_env_default_the_scoreboard_256() {
                let t0 = Instant::now();
                last = gpu
                    .decode_step(last)
                    .unwrap_or_else(|e| panic!("timed step at depth {depth}: {e:#}"));
                step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            print_plain_ctx_scaling_line(
                "qwen38-wgpu-mtp-depth",
                depth,
                step_ms.len(),
                warm_steps,
                &step_ms,
                &fill_label,
                &rev,
            );
        }
    }
}

#[cfg(feature = "cuda")]
fn argmax_host_row_lowest_index(row: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v.is_finite() && v > val {
            val = v;
            best = i;
        }
    }
    best as u32
}

#[cfg(feature = "cuda")]
const DEPTH_PROMPT_CORPUS_ENV_OPTS_IN_A_NONREPETITIVE_RAW_TEXT_PREFIX_SO_ACCEPTANCE_SITS_IN_THE_FREE_PROSE_BRACKET_INSTEAD_OF_THE_TILED_DOC_UPPER_BRACKET:
    &str = "NV_Q38_DEPTH_PROMPT_CORPUS";

#[cfg(feature = "cuda")]
fn tiled_prompt_ids(dir: &std::path::Path, depth: usize) -> Vec<u32> {
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    if let Ok(corpus) = std::env::var(
        DEPTH_PROMPT_CORPUS_ENV_OPTS_IN_A_NONREPETITIVE_RAW_TEXT_PREFIX_SO_ACCEPTANCE_SITS_IN_THE_FREE_PROSE_BRACKET_INSTEAD_OF_THE_TILED_DOC_UPPER_BRACKET,
    ) {
        if !corpus.is_empty() {
            let text = std::fs::read_to_string(&corpus)
                .unwrap_or_else(|e| panic!("read corpus prompt file {corpus}: {e}"));
            let ids: Vec<u32> = tok
                .encode(text.as_str(), false)
                .expect("encode corpus")
                .get_ids()
                .to_vec();
            assert!(
                ids.len() >= depth,
                "corpus {corpus} holds {} tokens but the rung needs {depth}; tiling it would \
                 re-create the repetition this arm exists to avoid",
                ids.len()
            );
            return ids[..depth].to_vec();
        }
    }
    let seed: Vec<u32> = tok
        .encode(
            TILED_PROMPT_PARAGRAPH_REPETITION_LANDS_IN_THE_HIGH_ACCEPT_DOC_REGIME_SO_ACCEPTANCE_HERE_IS_AN_UPPER_BRACKET_NOT_A_CHAT_NUMBER,
            false,
        )
        .expect("encode paragraph")
        .get_ids()
        .to_vec();
    assert!(!seed.is_empty(), "paragraph tokenized to nothing");
    (0..depth).map(|i| seed[i % seed.len()]).collect()
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 on the eager cuda dense arm plus the shipped \
            bf16 MTP head; set NV_QWEN38_SERVING_TEST=1 -- speculative (MTP-drafter) EFFECTIVE \
            decode tok/s (emitted tokens / wall) vs KV depth: the prompt is a paragraph tiled to \
            exactly the depth (REAL chunked prefill through Qwen38MtpDecodeSession::start, so \
            acceptance sits in the high-accept doc-repetition bracket, not the chat bracket); \
            NV_DRAFTER=mtp selects the mtp arm, anything else runs the plain eager \
            forward_with_cache control whose median must reconcile with the established \
            qwen38_cuda_dense_decode_ms_per_token eager rows; run ONE depth per process via \
            NV_CTX_TOKENS"]
fn qwen38_cuda_mtp_effective_decode_tok_s_vs_context_depth_real_chunked_prefill_eager_session_arm_keyed_on_nv_drafter(
) {
    use candle_core::{DType, Device, Tensor};
    use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
    use nv_specdecode::qwen38_mtp::{
        mtp_chain_depth_from_env, mtp_draft_dir_override_from_env, mtp_drafter_selected_from_env,
        Qwen38DenseMtpHead, Qwen38MtpDecodeSession, MTP_WEIGHTS_FILE_NAME,
    };

    require_gate("NV_QWEN38_SERVING_TEST");
    let mtp_arm = mtp_drafter_selected_from_env();
    let dir = match std::env::var("NV_QWEN38_DIR") {
        Ok(d) if !d.is_empty() => {
            let p = std::path::PathBuf::from(&d);
            assert!(p.is_dir(), "NV_QWEN38_DIR={d} is not a directory");
            p
        }
        _ => hub_dirs::snapshot(
            NVFP4_REPO,
            &["config.json", "tokenizer.json", MTP_WEIGHTS_FILE_NAME],
        )
        .unwrap_or_else(|| {
            panic!(
                "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no hydrated {NVFP4_REPO} \
                 snapshot carrying the MTP shard; set NV_QWEN38_DIR"
            )
        }),
    };
    let rev = snapshot_rev_label(&dir);
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = nv_weights::QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("track-2 dense NVFP4 loader must load Qwen3.8-27B before this ladder can run");
    drop(weights);
    let mtp = if mtp_arm {
        Some(
            Qwen38DenseMtpHead::from_checkpoint(
                mtp_draft_dir_override_from_env().as_deref(),
                &dir,
                &base,
                &device,
            )
            .expect("MTP head loads from the shipped model_mtp.safetensors"),
        )
    } else {
        None
    };
    let k = mtp_chain_depth_from_env();
    let max_pos = base.config().max_position_embeddings;

    let depths = ctx_tokens_from_env_default_256_8k_196k();
    for &depth in &depths {
        let max_seq = depth
            + KV_SLOT_HEADROOM_1024_COVERS_ANCHOR_WARMUP_TIMED_OVERSHOOT_AND_THE_512_ROW_VERIFY_GUARD;
        assert!(
            max_seq < max_pos,
            "a model declaring max_pos {max_pos} must serve depth {depth}, so a trip here is a \
             config bug"
        );
        let prompt = tiled_prompt_ids(&dir, depth);
        assert_eq!(prompt.len(), depth, "tiled prompt must be exactly the depth");

        if mtp_arm {
            let head = mtp.as_ref().expect("mtp arm holds the head");
            let t_pre = Instant::now();
            let mut session = Qwen38MtpDecodeSession::start(&base, head, k, &prompt, max_seq)
                .unwrap_or_else(|e| panic!("mtp session start at depth {depth}: {e:#}"));
            let prefill_s = t_pre.elapsed().as_secs_f64();
            assert_eq!(
                session.committed_len(),
                depth,
                "session prefill must commit exactly the prompt"
            );
            let (_warm_rounds, warm_emitted) = warmup_until_step_wall_plateaus(
                || {
                    assert!(session.round_fits(), "kv headroom exhausted during warmup");
                    session.round().expect("warm mtp round").len()
                },
                MTP_ARM_WARMUP_CAP_40_ROUNDS_BOUNDS_DEPTH_DRIFT_OF_THE_TIMED_WINDOW,
            );
            let rounds0 = session.stats.rounds;
            let drafted0 = session.stats.drafted;
            let accepted0 = session.stats.accepted;
            let draft_len_hist0 = session.submitted_draft_len_hist.clone();
            let accept_len_hist0 = session.stats.accept_len_hist.clone();
            let mut round_ms: Vec<f64> = Vec::new();
            let mut emitted = 0usize;
            let t0 = Instant::now();
            while emitted < timed_emitted_tokens_from_env_default_the_scoreboard_256() {
                assert!(
                    session.round_fits(),
                    "kv headroom exhausted mid-window at committed {}",
                    session.committed_len()
                );
                let r0 = Instant::now();
                emitted += session.round().expect("timed mtp round").len();
                round_ms.push(r0.elapsed().as_secs_f64() * 1e3);
            }
            let wall_s = t0.elapsed().as_secs_f64();
            let timed = RoundTally {
                rounds: session.stats.rounds - rounds0,
                drafted: session.stats.drafted - drafted0,
                accepted: session.stats.accepted - accepted0,
            };
            assert_eq!(
                session.committed_len(),
                depth + warm_emitted + emitted,
                "trunk commit must advance by exactly the emitted tokens"
            );
            assert_mtp_window_not_degenerate_then_print_ctx_scaling_line(
                "qwen38-cuda-mtp-depth",
                "mtp",
                depth,
                k,
                k,
                &timed,
                emitted,
                warm_emitted,
                wall_s,
                &round_ms,
                &format!("prefill_s={prefill_s:.1}"),
                &rev,
                &adaptive_shape_suffix_appended_after_checkpoint_so_parse_and_append_kv_parsing_stays_additive(
                    &p_min_label_matching_the_k_sweep_flag(),
                    &hist_window_delta_counts_only_grow(
                        &draft_len_hist0,
                        &session.submitted_draft_len_hist,
                    ),
                    &hist_window_delta_counts_only_grow(
                        &accept_len_hist0,
                        &session.stats.accept_len_hist,
                    ),
                ),
            );
        } else {
            let mut cache = base.new_kv_cache(max_seq).expect("kv cache");
            let chunk = nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
                .filter(|&c| depth > c)
                .unwrap_or(depth);
            let t_pre = Instant::now();
            let mut fed = 0usize;
            let mut last_row: Vec<f32> = Vec::new();
            while fed < depth {
                let n = chunk.min(depth - fed);
                let tokens =
                    Tensor::from_vec(prompt[fed..fed + n].to_vec(), (1usize, n), &device)
                        .expect("prompt chunk");
                let pos = Tensor::from_vec(
                    (fed as i32..(fed + n) as i32).collect::<Vec<i32>>(),
                    n,
                    &device,
                )
                .expect("positions");
                let (logits, _hidden) = base
                    .forward_with_cache_dispatched_hidden_rows(
                        &tokens,
                        &pos,
                        &mut cache,
                        None,
                        Some(1),
                    )
                    .unwrap_or_else(|e| panic!("chunked prefill at pos {fed}: {e:#}"));
                last_row = logits
                    .to_dtype(DType::F32)
                    .expect("f32")
                    .flatten_all()
                    .expect("flatten")
                    .to_vec1::<f32>()
                    .expect("sync to host");
                fed += n;
            }
            let prefill_s = t_pre.elapsed().as_secs_f64();
            assert_eq!(cache.current_len(), depth, "prefill must commit exactly the prompt");
            let mut last = argmax_host_row_lowest_index(&last_row);
            let mut step_pos = depth;
            let mut one_step = |last: &mut u32, step_pos: &mut usize| {
                let tokens = Tensor::from_vec(vec![*last], (1usize, 1usize), &device)
                    .expect("token");
                let positions =
                    Tensor::from_vec(vec![*step_pos as i32], 1usize, &device).expect("position");
                let logits = base
                    .forward_with_cache(&tokens, &positions, &mut cache)
                    .unwrap_or_else(|e| panic!("decode at pos {step_pos}: {e:#}"));
                let row = logits
                    .to_dtype(DType::F32)
                    .expect("f32")
                    .flatten_all()
                    .expect("flatten")
                    .to_vec1::<f32>()
                    .expect("sync to host");
                *last = argmax_host_row_lowest_index(&row);
                *step_pos += 1;
            };
            let (warm_steps, _warm_tokens) = warmup_until_step_wall_plateaus(
                || {
                    one_step(&mut last, &mut step_pos);
                    1
                },
                PLAIN_ARM_WARMUP_CAP_200_STEPS_MATCHES_THE_CTX_SCALING_LADDERS,
            );
            let mut step_ms: Vec<f64> = Vec::new();
            for _ in 0..timed_emitted_tokens_from_env_default_the_scoreboard_256() {
                let t0 = Instant::now();
                one_step(&mut last, &mut step_pos);
                step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            print_plain_ctx_scaling_line(
                "qwen38-cuda-mtp-depth",
                depth,
                step_ms.len(),
                warm_steps,
                &step_ms,
                &format!("prefill_s={prefill_s:.1}"),
                &rev,
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 once PER DEPTH into a fresh GraphedQwen3Moe \
            (the engine consumes the base by value and its KV cache binds max_seq at \
            construction, so a depth rung is a full engine rebuild sized depth+headroom); set \
            NV_QWEN38_SERVING_TEST=1 and NV_DRAFTER=mtp -- speculative (MTP-drafter) EFFECTIVE \
            decode tok/s (emitted tokens / wall) vs KV depth through \
            Qwen38MtpGraphedDecodeSession, the graphed engine behind the k-sweep shallow rows: \
            REAL chunked prefill of the same tiled paragraph the eager ladder feeds, then one \
            untimed round that records the rows-keyed verify graph (its wall lands in \
            capture_round0_ms, a serving-boot cost, never in the timed window); the eager arm in \
            this file is the control this arm's shallow rung must beat and its deep rungs are \
            the crossover question; run ONE depth per process via NV_CTX_TOKENS"]
fn qwen38_cuda_mtp_graphed_effective_decode_tok_s_vs_context_depth_real_chunked_prefill_per_depth_engine_rebuild(
) {
    use candle_core::Device;
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
    use nv_specdecode::qwen38_mtp::{
        mtp_chain_depth_from_env, mtp_draft_dir_override_from_env, mtp_drafter_selected_from_env,
        Qwen38DenseMtpHead, Qwen38MtpGraphedDecodeSession, MTP_WEIGHTS_FILE_NAME,
    };

    const EMITTED_IDENTITY_PREFIX_257_TOKENS_CAPTURE_PLUS_TIMED_GUARANTEE_IT_AND_A_DRAFTER_ONLY_CHANGE_MUST_NOT_MOVE_IT:
        usize = 257;
    fn fnv1a64_over_u32_le(tokens: &[u32]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for t in tokens {
            for b in t.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    require_gate("NV_QWEN38_SERVING_TEST");
    assert!(
        mtp_drafter_selected_from_env(),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: the graphed session is mtp-only, set \
         NV_DRAFTER=mtp so the recorded flags carry the full saturn spec env"
    );
    let dir = match std::env::var("NV_QWEN38_DIR") {
        Ok(d) if !d.is_empty() => {
            let p = std::path::PathBuf::from(&d);
            assert!(p.is_dir(), "NV_QWEN38_DIR={d} is not a directory");
            p
        }
        _ => hub_dirs::snapshot(
            NVFP4_REPO,
            &["config.json", "tokenizer.json", MTP_WEIGHTS_FILE_NAME],
        )
        .unwrap_or_else(|| {
            panic!(
                "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no hydrated {NVFP4_REPO} \
                 snapshot carrying the MTP shard; set NV_QWEN38_DIR"
            )
        }),
    };
    let rev = snapshot_rev_label(&dir);
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let k = mtp_chain_depth_from_env();

    let depths = ctx_tokens_from_env_default_256_8k_196k();
    for &depth in &depths {
        let max_seq = depth
            + KV_SLOT_HEADROOM_1024_COVERS_ANCHOR_WARMUP_TIMED_OVERSHOOT_AND_THE_512_ROW_VERIFY_GUARD;
        let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
        assert!(
            max_seq < cfg.max_position_embeddings,
            "a model declaring max_pos {} must serve depth {depth}, so a trip here is a config \
             bug",
            cfg.max_position_embeddings
        );
        let prompt = tiled_prompt_ids(&dir, depth);
        assert_eq!(prompt.len(), depth, "tiled prompt must be exactly the depth");

        let qcfg = nv_weights::QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
        let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("weights");
        let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
            .expect("track-2 dense NVFP4 loader must load Qwen3.8-27B before this ladder can run");
        drop(weights);
        let t_build = Instant::now();
        let mut engine = GraphedQwen3Moe::new(base, &device, max_seq)
            .unwrap_or_else(|e| panic!("graphed engine at max_seq {max_seq}: {e:#}"));
        let engine_build_s = t_build.elapsed().as_secs_f64();
        let mtp = Qwen38DenseMtpHead::from_checkpoint(
            mtp_draft_dir_override_from_env().as_deref(),
            &dir,
            engine.underlying(),
            &device,
        )
        .expect("MTP head loads from the shipped model_mtp.safetensors");

        let t_pre = Instant::now();
        let mut session = Qwen38MtpGraphedDecodeSession::start(&mut engine, &mtp, k, &prompt)
            .unwrap_or_else(|e| panic!("graphed mtp session start at depth {depth}: {e:#}"));
        let prefill_s = t_pre.elapsed().as_secs_f64();
        assert_eq!(
            session.committed_len(),
            depth,
            "session prefill must commit exactly the prompt"
        );

        assert!(session.round_fits(), "kv headroom exhausted before the capture round");
        let mut emitted_stream: Vec<u32> = Vec::new();
        let t_cap = Instant::now();
        let capture_round = session
            .round()
            .expect("capture round: the first round records the rows-keyed verify graph");
        let capture_round0_ms = t_cap.elapsed().as_secs_f64() * 1e3;
        emitted_stream.extend(&capture_round);
        let capture_emitted = capture_round.len();

        let (_warm_rounds, warm_emitted) = warmup_until_step_wall_plateaus(
            || {
                assert!(session.round_fits(), "kv headroom exhausted during warmup");
                let r = session.round().expect("warm graphed mtp round");
                emitted_stream.extend(&r);
                r.len()
            },
            MTP_ARM_WARMUP_CAP_40_ROUNDS_BOUNDS_DEPTH_DRIFT_OF_THE_TIMED_WINDOW,
        );
        let rounds0 = session.stats.rounds;
        let drafted0 = session.stats.drafted;
        let accepted0 = session.stats.accepted;
        let draft_len_hist0 = session.submitted_draft_len_hist.clone();
        let accept_len_hist0 = session.stats.accept_len_hist.clone();
        let draft_ms0 = session.stats.draft_ms;
        let verify_ms0 = session.stats.verify_ms;
        let commit_ms0 = session.stats.commit_ms;
        let mut round_ms: Vec<f64> = Vec::new();
        let mut emitted = 0usize;
        let t0 = Instant::now();
        while emitted < timed_emitted_tokens_from_env_default_the_scoreboard_256() {
            assert!(
                session.round_fits(),
                "kv headroom exhausted mid-window at committed {}",
                session.committed_len()
            );
            let r0 = Instant::now();
            let r = session.round().expect("timed graphed mtp round");
            round_ms.push(r0.elapsed().as_secs_f64() * 1e3);
            emitted_stream.extend(&r);
            emitted += r.len();
        }
        let wall_s = t0.elapsed().as_secs_f64();
        let timed = RoundTally {
            rounds: session.stats.rounds - rounds0,
            drafted: session.stats.drafted - drafted0,
            accepted: session.stats.accepted - accepted0,
        };
        let timed_rounds_f = timed.rounds.max(1) as f64;
        eprintln!(
            "CTX-PHASES qwen38-cuda-mtp-depth depth={depth} arm=mtp-graphed rounds={} \
             draft_ms_round={:.2} verify_ms_round={:.2} commit_ms_round={:.2} \
             phase_sum_ms_round={:.2} round_ms_median={:.1}",
            timed.rounds,
            (session.stats.draft_ms - draft_ms0) / timed_rounds_f,
            (session.stats.verify_ms - verify_ms0) / timed_rounds_f,
            (session.stats.commit_ms - commit_ms0) / timed_rounds_f,
            (session.stats.draft_ms - draft_ms0 + session.stats.verify_ms - verify_ms0
                + session.stats.commit_ms
                - commit_ms0)
                / timed_rounds_f,
            median_of(&round_ms),
        );
        let prefix =
            EMITTED_IDENTITY_PREFIX_257_TOKENS_CAPTURE_PLUS_TIMED_GUARANTEE_IT_AND_A_DRAFTER_ONLY_CHANGE_MUST_NOT_MOVE_IT;
        assert!(
            emitted_stream.len() >= prefix,
            "graphed arm emitted {} tokens, below the {prefix}-token identity prefix",
            emitted_stream.len()
        );
        eprintln!(
            "CTX-EMITTED qwen38-cuda-mtp-depth depth={depth} arm=mtp-graphed \
             prefix_tokens={prefix} fnv1a64={:016x} head8={:?}",
            fnv1a64_over_u32_le(&emitted_stream[..prefix]),
            &emitted_stream[..8.min(emitted_stream.len())],
        );
        assert_eq!(
            session.committed_len(),
            depth + capture_emitted + warm_emitted + emitted,
            "trunk commit must advance by exactly the emitted tokens"
        );
        if let Some((s_rounds, s_drafted, s_accepted)) = session.suffix_rounds_drafted_accepted() {
            eprintln!(
                "CTX-SUFFIX qwen38-cuda-mtp-depth depth={depth} arm=mtp-graphed \
                 suffix_rounds={s_rounds} suffix_drafted={s_drafted} suffix_accepted={s_accepted} \
                 total_rounds={} suffix_accept_rate={:.3}",
                session.stats.rounds,
                if s_drafted > 0 {
                    s_accepted as f64 / s_drafted as f64
                } else {
                    f64::NAN
                },
            );
        }
        if session.tree_rounds > 0 {
            eprintln!(
                "CTX-TREE qwen38-cuda-mtp-depth depth={depth} arm=mtp-graphed tree_rounds={} \
                 branch_b_rounds={} branch_b_recovered_tokens={} recovery_rate={:.4}",
                session.tree_rounds,
                session.tree_branch_b_rounds,
                session.tree_branch_b_recovered_tokens,
                session.tree_branch_b_rounds as f64 / session.tree_rounds.max(1) as f64,
            );
        }
        let max_round_overshoot = if session.suffix_rounds_drafted_accepted().is_some() {
            nv_specdecode::qwen38_sufspec::suffix_draft_max_from_env()
        } else {
            k
        };
        assert_mtp_window_not_degenerate_then_print_ctx_scaling_line(
            "qwen38-cuda-mtp-depth",
            "mtp-graphed",
            depth,
            k,
            max_round_overshoot,
            &timed,
            emitted,
            warm_emitted,
            wall_s,
            &round_ms,
            &format!(
                "prefill_s={prefill_s:.1} engine_build_s={engine_build_s:.1} \
                 capture_round0_ms={capture_round0_ms:.1} max_seq_allocated={max_seq}"
            ),
            &rev,
            &adaptive_shape_suffix_appended_after_checkpoint_so_parse_and_append_kv_parsing_stays_additive(
                &p_min_label_matching_the_k_sweep_flag(),
                &hist_window_delta_counts_only_grow(
                    &draft_len_hist0,
                    &session.submitted_draft_len_hist,
                ),
                &hist_window_delta_counts_only_grow(
                    &accept_len_hist0,
                    &session.stats.accept_len_hist,
                ),
            ),
        );
    }
}
