#![cfg(feature = "cuda")]

use candle_core::Device;
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
use nv_models::qwen3_5_mtp::{MtpHead, MtpMoeEagerDecodeSession, MtpMoeGraphedDecodeSession};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const NEW_TOKENS_128_ENOUGH_FOR_A_STEADY_RATE: usize = 128;
const BASELINE_WARMUP_STEPS_16_ABSORB_GRAPH_CAPTURE_AND_CLOCK_RAMP: usize = 16;
const K_SWEEP_DESCENDING_SO_THE_LARGEST_VERIFY_LANE_ALLOCATES_ONCE: [usize; 4] = [5, 4, 3, 2];
const STORY_SEED: &str = "The lighthouse keeper had counted nine hundred and twelve storms from \
    the gallery deck, and he wrote each one into the ledger with the same steel pen. On the \
    morning the sea turned the color of old brass, he found a rowboat lodged in the rocks below \
    the light, its oars shipped neatly, its hull dry on the inside. There was no name on the \
    stern. There was a brass key on the center thwart, and under the key a note in his own \
    handwriting, dated forty years ahead. ";

fn snapshot_with_mtp_shard() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN36_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("qwen3.6 snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file() && p.join("model_mtp.safetensors").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no qwen3.6 snapshot carrying model_mtp.safetensors; set NV_QWEN36_DIR")
}

const FACTUAL_SEED_CHANGELOG_TILING_TARGETS_THE_LLAMA_CPP_92_96_PERCENT_ACCEPT_REGIME: &str =
    "## 1.4.1\n- Fixed a crash when the config file is missing.\n- Improved startup time by \
     caching the model index.\n\n## 1.4.2\n- Fixed a crash when the audio device is \
     unplugged.\n- Improved startup time by lazy-loading the tokenizer.\n\n## 1.4.3\n- Fixed a \
     crash when the network is unreachable.\n- Improved startup time by deferring the license \
     check.\n\n";

const CODE_SEED_REPETITIVE_HELPERS_SIT_BETWEEN_THE_DOC_AND_CREATIVE_ACCEPT_BRACKETS: &str =
    "def add_scaled(a, b, scale):\n    return a + b * scale\n\ndef sub_scaled(a, b, scale):\n    \
     return a - b * scale\n\ndef mul_scaled(a, b, scale):\n    return a * b * scale\n\ndef \
     clamp_scaled(a, lo, hi, scale):\n    return min(max(a * scale, lo), hi)\n\n";

fn prompt_of_depth_with(tok: &Tokenizer, depth: usize, seed_text: &str, ask: &str) -> Vec<u32> {
    prompt_of_depth_raw_tail(
        tok,
        depth,
        seed_text,
        &format!("\n\n{ask}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    )
}

fn prompt_of_depth_raw_tail(
    tok: &Tokenizer,
    depth: usize,
    seed_text: &str,
    tail_text: &str,
) -> Vec<u32> {
    let enc = |s: &str| -> Vec<u32> {
        tok.encode(s, false).expect("encode").get_ids().to_vec()
    };
    let head = enc("<|im_start|>user\n");
    let tail = enc(tail_text);
    let seed = enc(seed_text);
    let ctx_needed = depth
        .checked_sub(head.len() + tail.len())
        .expect("depth must fit the template frame");
    let mut out = head;
    let mut ctx: Vec<u32> = Vec::with_capacity(ctx_needed + seed.len());
    while ctx.len() < ctx_needed {
        ctx.extend_from_slice(&seed);
    }
    ctx.truncate(ctx_needed);
    out.extend_from_slice(&ctx);
    out.extend_from_slice(&tail);
    assert_eq!(out.len(), depth);
    out
}

fn prompt_of_depth(tok: &Tokenizer, depth: usize) -> Vec<u32> {
    prompt_of_depth_with(
        tok,
        depth,
        STORY_SEED,
        "Continue this story in the same style.",
    )
}

fn class_prompt_of_depth(tok: &Tokenizer, class: &str, depth: usize) -> Vec<u32> {
    match class {
        "factual" => prompt_of_depth_with(
            tok,
            depth,
            FACTUAL_SEED_CHANGELOG_TILING_TARGETS_THE_LLAMA_CPP_92_96_PERCENT_ACCEPT_REGIME,
            "Continue this changelog with more entries in exactly the same format.",
        ),
        "creative" => prompt_of_depth(tok, depth),
        "code" => prompt_of_depth_with(
            tok,
            depth,
            CODE_SEED_REPETITIVE_HELPERS_SIT_BETWEEN_THE_DOC_AND_CREATIVE_ACCEPT_BRACKETS,
            "Continue this file with more helper functions in exactly the same style.",
        ),
        "thinking" => prompt_of_depth_raw_tail(
            tok,
            depth,
            STORY_SEED,
            "\n\nWhat is the capital of France, and why did it become the capital? Answer in a \
             few sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n",
        ),
        "copy" => prompt_of_depth_with(
            tok,
            depth,
            "The quick brown fox jumps over the lazy dog while the calm gray cat watches from \
             the warm stone wall near the old wooden gate. ",
            "Copy the text above word for word, exactly as written.",
        ),
        other => panic!("unknown prompt class {other:?}"),
    }
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i as u32;
        }
    }
    best
}

fn chunked_prefill_last_row_plain_primed_then_grouped_installed(
    engine: &mut GraphedQwen3Moe,
    prompt: &[u32],
) -> Vec<f32> {
    engine.reset().expect("engine reset");
    let stashed = engine
        .take_moe_dispatch_for_host_routed_prefill_then_restore_reinstalling_would_drop_and_rebuild_the_folded_weights_a_torn_down_graph_baked();
    let mut fed = 0usize;
    let mut last = Vec::new();
    while fed < prompt.len() {
        let n = 512.min(prompt.len() - fed);
        last = engine
            .prefill(&prompt[fed..fed + n])
            .unwrap_or_else(|e| panic!("prefill chunk at {fed}: {e:#}"));
        fed += n;
    }
    match stashed {
        Some(hook) => {
            engine.set_moe_dispatch(Some(hook));
            engine.set_device_routing(true);
        }
        None => engine.install_grouped_moe().expect("grouped install"),
    }
    last
}

fn eager_baseline_ms_per_tok(engine: &GraphedQwen3Moe, prompt: &[u32]) -> f64 {
    use candle_core::Tensor;
    let base = engine.underlying();
    let disp = engine.moe_dispatch_ref();
    let device = base.device().clone();
    let max_seq = prompt.len()
        + NEW_TOKENS_128_ENOUGH_FOR_A_STEADY_RATE
        + BASELINE_WARMUP_STEPS_16_ABSORB_GRAPH_CAPTURE_AND_CLOCK_RAMP
        + 2;
    let mut cache = base.new_kv_cache(max_seq).expect("eager cache");
    let seq = prompt.len();
    let tokens = Tensor::from_vec(prompt.to_vec(), (1usize, seq), &device).expect("tokens");
    let pos = Tensor::from_vec((0..seq as i32).collect::<Vec<i32>>(), seq, &device).expect("pos");
    let logits = base
        .forward_with_cache_dispatched(
            &tokens,
            &pos,
            &mut cache,
            nv_models::qwen3_5_mtp::PREFILL_ROUTES_HOST_BECAUSE_THE_GROUPED_MULTI_TOKEN_COST_EXPLODES_WITH_M,
        )
        .expect("eager prefill");
    let last: Vec<f32> = logits
        .narrow(1, seq - 1, 1)
        .expect("row")
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flat")
        .to_vec1()
        .expect("host");
    drop(logits);
    let mut cur = argmax(&last);
    let step = |cur: &mut u32, cache: &mut nv_models::qwen3_5_moe::Qwen3MoeKvCache| {
        let l = cache.current_len();
        let t = Tensor::from_vec(vec![*cur], (1usize, 1usize), &device).expect("t");
        let p = Tensor::from_vec(vec![l as i32], 1usize, &device).expect("p");
        let lg = base
            .forward_with_cache_dispatched(&t, &p, cache, disp)
            .expect("eager decode");
        let row: Vec<f32> = lg
            .to_dtype(candle_core::DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flat")
            .to_vec1()
            .expect("host");
        *cur = argmax(&row);
    };
    for _ in 0..BASELINE_WARMUP_STEPS_16_ABSORB_GRAPH_CAPTURE_AND_CLOCK_RAMP {
        step(&mut cur, &mut cache);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..NEW_TOKENS_128_ENOUGH_FOR_A_STEADY_RATE {
        step(&mut cur, &mut cache);
    }
    1000.0 * t0.elapsed().as_secs_f64() / NEW_TOKENS_128_ENOUGH_FOR_A_STEADY_RATE as f64
}

fn baseline_ms_per_tok(engine: &mut GraphedQwen3Moe, prompt: &[u32]) -> (f64, usize) {
    let last = chunked_prefill_last_row_plain_primed_then_grouped_installed(engine, prompt);
    let nan = last.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan, 0, "baseline prefill produced {nan} NaN logits");
    let mut cur = argmax(&last);
    for _ in 0..BASELINE_WARMUP_STEPS_16_ABSORB_GRAPH_CAPTURE_AND_CLOCK_RAMP {
        engine.forward_decode(cur).expect("warm decode");
        cur = argmax(&engine.logits_host().expect("warm logits"));
    }
    let t0 = std::time::Instant::now();
    let mut steps = 0usize;
    for _ in 0..NEW_TOKENS_128_ENOUGH_FOR_A_STEADY_RATE {
        engine.forward_decode(cur).expect("decode");
        cur = argmax(&engine.logits_host().expect("logits"));
        steps += 1;
    }
    let ms = 1000.0 * t0.elapsed().as_secs_f64() / steps as f64;
    (ms, steps)
}

struct SpecOut {
    steady_ms_per_tok: f64,
    accept_rate: f64,
    tokens_per_round: f64,
    rounds: usize,
    new_tokens: usize,
    draft_ms_avg: f64,
    verify_ms_avg: f64,
    submitted_hist: String,
}

fn hist_string(hist: &std::collections::BTreeMap<usize, usize>) -> String {
    let parts: Vec<String> = hist.iter().map(|(len, n)| format!("{len}:{n}")).collect();
    parts.join(",")
}

fn absorb_the_one_shot_cuda_invalid_value_a_captured_decode_teardown_leaves(
    engine: &mut GraphedQwen3Moe,
) {
    for attempt in 0..2 {
        match engine.reset() {
            Ok(()) => return,
            Err(e) => eprintln!("[econ] reset attempt {attempt} absorbed: {e:#}"),
        }
    }
    panic!("engine reset failed twice; the context is genuinely poisoned");
}

fn graphed_spec_ms_per_tok(
    engine: &mut GraphedQwen3Moe,
    mtp: &MtpHead,
    k: usize,
    prompt: &[u32],
) -> SpecOut {
    absorb_the_one_shot_cuda_invalid_value_a_captured_decode_teardown_leaves(engine);
    let t_start = std::time::Instant::now();
    eprintln!("[econ-progress] graphed session start: prompt={} k={k}", prompt.len());
    let mut session = MtpMoeGraphedDecodeSession::start(engine, mtp, k, prompt).expect("session");
    eprintln!(
        "[econ-progress] graphed session started in {:.1}s",
        t_start.elapsed().as_secs_f64()
    );
    let mut new_tokens = 1usize;
    let mut per_round_tokens: Vec<usize> = Vec::new();
    while new_tokens < NEW_TOKENS_128_ENOUGH_FOR_A_STEADY_RATE && session.round_fits() {
        let emitted = session.round().expect("round");
        per_round_tokens.push(emitted.len());
        new_tokens += emitted.len();
    }
    let submitted_hist = hist_string(&session.submitted_draft_len_hist);
    finish_spec(
        session.stats.clone(),
        per_round_tokens,
        new_tokens,
        submitted_hist,
    )
}

fn eager_spec_ms_per_tok(
    engine: &GraphedQwen3Moe,
    mtp: &MtpHead,
    k: usize,
    prompt: &[u32],
) -> SpecOut {
    let max_seq = prompt.len() + NEW_TOKENS_128_ENOUGH_FOR_A_STEADY_RATE + k + 2;
    let t_start = std::time::Instant::now();
    eprintln!("[econ-progress] eager session start: prompt={} k={k}", prompt.len());
    let mut session = MtpMoeEagerDecodeSession::start(
        engine.underlying(),
        mtp,
        k,
        prompt,
        max_seq,
        engine.moe_dispatch_ref(),
    )
    .expect("eager session");
    eprintln!(
        "[econ-progress] eager session started in {:.1}s",
        t_start.elapsed().as_secs_f64()
    );
    let mut new_tokens = 1usize;
    let mut per_round_tokens: Vec<usize> = Vec::new();
    while new_tokens < NEW_TOKENS_128_ENOUGH_FOR_A_STEADY_RATE && session.round_fits() {
        let emitted = session.round().expect("round");
        per_round_tokens.push(emitted.len());
        new_tokens += emitted.len();
    }
    let submitted_hist = hist_string(&session.submitted_draft_len_hist);
    finish_spec(
        session.stats.clone(),
        per_round_tokens,
        new_tokens,
        submitted_hist,
    )
}

fn finish_spec(
    stats: nv_models::qwen3_5_mtp::MtpSpecStats,
    per_round_tokens: Vec<usize>,
    new_tokens: usize,
    submitted_hist: String,
) -> SpecOut {
    assert_eq!(stats.round_ms.len(), per_round_tokens.len());
    let steady_rounds = stats.round_ms.len().saturating_sub(1);
    assert!(
        steady_rounds >= 8,
        "only {steady_rounds} steady rounds; the rate below would be noise"
    );
    let steady_ms: f64 = stats.round_ms[1..].iter().sum();
    let steady_toks: usize = per_round_tokens[1..].iter().sum();
    SpecOut {
        steady_ms_per_tok: steady_ms / steady_toks as f64,
        accept_rate: stats.accept_rate(),
        tokens_per_round: stats.tokens_per_round(),
        rounds: stats.rounds,
        new_tokens,
        draft_ms_avg: stats.draft_ms / stats.rounds as f64,
        verify_ms_avg: stats.verify_ms / stats.rounds as f64,
        submitted_hist,
    }
}

#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B checkpoint; set NV_QWEN36_MTP_ECON=1 and run in \
            --release under the GPU lock -- same-process interleaved spec-on (MoE MTP, graphed \
            m=k+1 verify lane) vs spec-off (the record graphed grouped-MoE m=1 decode) at \
            context depths 256 and 8192, k swept 5..2; steady-state excludes each arm's first \
            round/warmup so no graph capture is billed to the rate"]
fn qwen36_mtp_econ_spec_on_vs_off_at_256_and_8k() {
    if std::env::var("NV_QWEN36_MTP_ECON").as_deref() != Ok("1") {
        panic!("set NV_QWEN36_MTP_ECON=1 to run (it must never silently skip)");
    }
    let dir = snapshot_with_mtp_shard();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw).expect("config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);
    let mtp = MtpHead::from_safetensors(dir.join("model_mtp.safetensors"), &model, &device)
        .expect("MoE MTP head");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");

    let max_depth = 8192usize;
    let slots = max_depth + NEW_TOKENS_128_ENOUGH_FOR_A_STEADY_RATE + 64;
    let mut engine = GraphedQwen3Moe::new(model, &device, slots).expect("graphed engine");
    engine.install_grouped_moe().expect("grouped install");

    let snap = dir
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    eprintln!(
        "[q36-mtp-econ] basis: checkpoint=RedHatAI/Qwen3.6-35B-A3B-NVFP4@{snap} backend=cuda \
         engine=GraphedQwen3Moe grouped-moe graph decode (record arm) vs MoE-MTP graphed \
         verify-lane sessions, greedy, bs=1, {} new tokens per run, story-continuation prompt \
         through the im_start template, steady-state excludes round/step 0",
        NEW_TOKENS_128_ENOUGH_FOR_A_STEADY_RATE
    );

    if let Ok(d) = std::env::var("NV_Q36_ECON_CAPTURED_BASELINE_ONLY_DEPTH") {
        let d: usize = d.parse().expect("NV_Q36_ECON_CAPTURED_BASELINE_ONLY_DEPTH");
        let prompt = prompt_of_depth(&tok, d);
        let (ms, _) = baseline_ms_per_tok(&mut engine, &prompt);
        eprintln!(
            "ECON depth={d} arm=spec-off pass=solo ms_per_tok={ms:.3} capture_active={} \
             one_captured_decode_per_process_because_a_reset_after_the_decode_fix_capture_\
             dies_with_cuda_invalid_value",
            engine.capture_active()
        );
        return;
    }

    let ks: Vec<usize> = match std::env::var("NV_Q36_ECON_KS") {
        Ok(v) => v
            .split(',')
            .map(|s| s.trim().parse::<usize>().expect("NV_Q36_ECON_KS entries are k values"))
            .collect(),
        Err(_) => K_SWEEP_DESCENDING_SO_THE_LARGEST_VERIFY_LANE_ALLOCATES_ONCE.to_vec(),
    };
    assert!(
        ks.windows(2).all(|w| w[0] > w[1]) && !ks.is_empty() && ks[ks.len() - 1] >= 1,
        "NV_Q36_ECON_KS must be strictly descending so the largest verify lane allocates once: \
         {ks:?}"
    );

    let prompts: Vec<(usize, Vec<u32>)> = [256usize, 8192]
        .iter()
        .map(|&d| {
            let p = prompt_of_depth(&tok, d);
            assert_eq!(p.len(), d);
            (d, p)
        })
        .collect();
    eprintln!("[econ-progress] prompts built: {:?}", prompts.iter().map(|(d, _)| *d).collect::<Vec<_>>());

    for (depth, prompt) in &prompts {
        let depth = *depth;
        for &k in &ks {
            let g = graphed_spec_ms_per_tok(&mut engine, &mtp, k, prompt);
            eprintln!(
                "ECON depth={depth} arm=spec-graphed-verify k={k} ms_per_tok={:.3} \
                 accept={:.3} tok_per_round={:.2} rounds={} new_toks={} draft_ms_avg={:.2} \
                 verify_ms_avg={:.2}",
                g.steady_ms_per_tok,
                g.accept_rate,
                g.tokens_per_round,
                g.rounds,
                g.new_tokens,
                g.draft_ms_avg,
                g.verify_ms_avg,
            );
            assert!(
                g.accept_rate > 0.0,
                "depth {depth} k={k}: zero acceptance on the graphed-verify arm, the \
                 economics above are for a broken drafter"
            );
            let e = eager_spec_ms_per_tok(&engine, &mtp, k, prompt);
            eprintln!(
                "ECON depth={depth} arm=spec-eager k={k} ms_per_tok={:.3} accept={:.3} \
                 tok_per_round={:.2} rounds={} new_toks={} draft_ms_avg={:.2} \
                 verify_ms_avg={:.2}",
                e.steady_ms_per_tok,
                e.accept_rate,
                e.tokens_per_round,
                e.rounds,
                e.new_tokens,
                e.draft_ms_avg,
                e.verify_ms_avg,
            );
            assert!(
                e.accept_rate > 0.0,
                "depth {depth} k={k}: zero acceptance on the eager serving arm, the \
                 economics above are for a broken drafter"
            );
        }

        absorb_the_one_shot_cuda_invalid_value_a_captured_decode_teardown_leaves(&mut engine);
        let eager_base_ms = eager_baseline_ms_per_tok(&engine, prompt);
        eprintln!("ECON depth={depth} arm=spec-off-eager ms_per_tok={eager_base_ms:.3}");
    }

    let (depth, prompt) = &prompts[0];
    let (base_ms, _) = baseline_ms_per_tok(&mut engine, prompt);
    eprintln!(
        "ECON depth={depth} arm=spec-off pass=1 ms_per_tok={base_ms:.3} capture_active={} \
         remaining_captured_baselines_run_solo_via_NV_Q36_ECON_CAPTURED_BASELINE_ONLY_DEPTH_\
         because_a_reset_after_the_decode_fix_capture_dies_with_cuda_invalid_value",
        engine.capture_active()
    );
}

fn usize_list_env(name: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(name) {
        Ok(v) => v
            .split(',')
            .map(|s| s.trim().parse::<usize>().unwrap_or_else(|_| panic!("{name} entries: {v:?}")))
            .collect(),
        Err(_) => default.to_vec(),
    }
}

fn f32_list_env(name: &str, default: &[f32]) -> Vec<f32> {
    match std::env::var(name) {
        Ok(v) => v
            .split(',')
            .map(|s| s.trim().parse::<f32>().unwrap_or_else(|_| panic!("{name} entries: {v:?}")))
            .collect(),
        Err(_) => default.to_vec(),
    }
}

fn string_list_env(name: &str, default: &[&str]) -> Vec<String> {
    match std::env::var(name) {
        Ok(v) => v.split(',').map(|s| s.trim().to_string()).collect(),
        Err(_) => default.iter().map(|s| s.to_string()).collect(),
    }
}

#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B checkpoint; set NV_QWEN36_MTP_ECON_P_MIN=1 and run in \
            --release under the GPU lock -- same-process interleaved fixed-k vs adaptive \
            NV_Q36_MTP_P_MIN sessions on the graphed verify-lane arm, per prompt class \
            (factual/creative/code) at depths 256 and 8192; the harness owns the p_min env per \
            arm; steady-state excludes each session's round 0 and the adaptive sessions \
            precapture every 2..=k+1 verify shape at start so no mid-run capture is billed"]
fn qwen36_mtp_econ_adaptive_p_min_vs_fixed_k_across_prompt_classes() {
    if std::env::var("NV_QWEN36_MTP_ECON_P_MIN").as_deref() != Ok("1") {
        panic!("set NV_QWEN36_MTP_ECON_P_MIN=1 to run (it must never silently skip)");
    }
    assert!(
        std::env::var("NV_Q36_MTP_P_MIN").is_err(),
        "unset NV_Q36_MTP_P_MIN: this harness owns the p_min env per arm, an ambient value \
         would turn the fixed-k baselines adaptive"
    );
    let depths = usize_list_env("NV_Q36_PMIN_DEPTHS", &[256, 8192]);
    let classes = string_list_env("NV_Q36_PMIN_CLASSES", &["factual", "creative", "code"]);
    let fixed_ks = usize_list_env("NV_Q36_PMIN_FIXED_KS", &[3, 2]);
    let adaptive_ks = usize_list_env("NV_Q36_PMIN_ADAPTIVE_KS", &[4, 3]);
    let p_sweep = f32_list_env("NV_Q36_PMIN_SWEEP", &[0.6, 0.75, 0.85]);

    let dir = snapshot_with_mtp_shard();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw).expect("config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);
    let mtp = MtpHead::from_safetensors(dir.join("model_mtp.safetensors"), &model, &device)
        .expect("MoE MTP head");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");

    let max_depth = depths.iter().copied().max().expect("at least one depth");
    let slots = max_depth + NEW_TOKENS_128_ENOUGH_FOR_A_STEADY_RATE + 64;
    let mut engine = GraphedQwen3Moe::new(model, &device, slots).expect("graphed engine");
    engine.install_grouped_moe().expect("grouped install");

    let snap = dir
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    eprintln!(
        "[q36-pmin-econ] basis: checkpoint=RedHatAI/Qwen3.6-35B-A3B-NVFP4@{snap} backend=cuda \
         engine=GraphedQwen3Moe graphed verify-lane MoE-MTP sessions, greedy, bs=1, {} new \
         tokens per arm, depth-exact templated class prompts, steady-state excludes round 0, \
         all arms same process",
        NEW_TOKENS_128_ENOUGH_FOR_A_STEADY_RATE
    );

    let report = |arm: &str, class: &str, depth: usize, k: usize, p_flag: &str, g: &SpecOut| {
        eprintln!(
            "ECON-PMIN depth={depth} class={class} arm={arm} k={k} p_min={p_flag} \
             ms_per_tok={:.3} accept={:.3} tok_per_round={:.2} rounds={} new_toks={} \
             draft_ms_avg={:.2} verify_ms_avg={:.2} submitted_hist={}",
            g.steady_ms_per_tok,
            g.accept_rate,
            g.tokens_per_round,
            g.rounds,
            g.new_tokens,
            g.draft_ms_avg,
            g.verify_ms_avg,
            g.submitted_hist,
        );
        assert!(
            g.accept_rate > 0.0,
            "depth {depth} class {class} arm {arm} k={k} p_min={p_flag}: zero acceptance, the \
             economics above are for a broken drafter"
        );
    };

    for &depth in &depths {
        for class in &classes {
            let prompt = class_prompt_of_depth(&tok, class, depth);
            assert_eq!(prompt.len(), depth);
            for &k in &adaptive_ks {
                for &p in &p_sweep {
                    std::env::set_var("NV_Q36_MTP_P_MIN", format!("{p}"));
                    let g = graphed_spec_ms_per_tok(&mut engine, &mtp, k, &prompt);
                    std::env::remove_var("NV_Q36_MTP_P_MIN");
                    report("adaptive", class, depth, k, &format!("{p:.2}"), &g);
                }
            }
            for &k in &fixed_ks {
                let g = graphed_spec_ms_per_tok(&mut engine, &mtp, k, &prompt);
                report("fixed", class, depth, k, "off", &g);
            }
        }
    }
}
