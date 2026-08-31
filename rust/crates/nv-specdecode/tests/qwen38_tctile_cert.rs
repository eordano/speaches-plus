#![cfg(feature = "cuda")]

mod hub_dirs;
mod sufspec_classes;

#[path = "../../nv-models/tests/ppl_common/mod.rs"]
#[allow(dead_code)]
mod ppl_common;

use candle_core::{DType, Device, Tensor};
use ppl_common::TeacherForcedNllFp32SoftmaxF64Sum;
use std::time::Instant;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const CHECKPOINT_LABEL: &str = "unsloth/Qwen3.8-27B-NVFP4";

const CERT_DEPTH_DEFAULT_8192_THE_SHARED_KNEE_THE_TCTILE_GATE_ADOPTION_MOTION_TARGETS: usize = 8192;
const CERT_EMITTED_TOKENS_512_MATCHES_THE_CHAMPION_LADDER_TIMED_WINDOW: usize = 512;
const CERT_KV_HEADROOM_1024_COVERS_ANCHOR_AND_ROUND_OVERSHOOT: usize = 1024;
const CERT_WALK_CONTINUATION_TOKENS_2048_GIVES_511_M4_VERIFY_ROUNDS_PER_ARM: usize = 2048;
const CERT_WALK_ROWS_M4_THE_CHAIN_SHAPE_THE_K3_ROUTE_SERVES: usize = 4;
const CERT_PREFILL_CHUNK_512_MATCHES_THE_TCTILE_GATE_FEED: usize = 512;

const TOPK_8_SCORES_KEPT_PER_VERIFY_ROW_SO_A_FLIPPED_TOKENS_SCORE_IN_THE_OTHER_ARM_IS_ON_HAND:
    usize = 8;

const NLL_DELTA_ABS_FLOOR_0_02_NATS_PLUS_2_PCT_RELATIVE_BECAUSE_ROUTER_TIE_FLIPS_MOVE_SINGLE_POSITIONS_BY_WHOLE_NATS_WITHOUT_SYSTEMATIC_DAMAGE:
    f64 = 0.02;
const NLL_DELTA_REL_2_PCT_OF_THE_REFERENCE_ARMS_MEAN_NLL: f64 = 0.02;
const NLL_DELTA_CROSS_CLASS_MEAN_BAND_0_02_NATS_A_ONE_SIDED_SHIFT_ON_EVERY_CLASS_IS_DAMAGE_NOT_TIE_NOISE:
    f64 = 0.02;
const TOP1_ACC_DELTA_BAND_1_PCT_ON_MATCHED_FORCED_TEXT: f64 = 0.01;

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
    }
}

fn cert_depth_from_env_single_value_default_8192() -> usize {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => {
            let s = v.trim();
            assert!(
                !s.contains(','),
                "NV_CTX_TOKENS={v}: the cert suite runs one depth per process"
            );
            let (num, mult) = match s.strip_suffix(['k', 'K']) {
                Some(n) => (n, 1024usize),
                None => (s, 1usize),
            };
            num.parse::<usize>().expect("NV_CTX_TOKENS") * mult
        }
        Err(_) => CERT_DEPTH_DEFAULT_8192_THE_SHARED_KNEE_THE_TCTILE_GATE_ADOPTION_MOTION_TARGETS,
    }
}

fn cert_emitted_tokens_from_env_default_512() -> usize {
    match std::env::var("NV_TIMED_EMITTED_TOKENS") {
        Ok(v) => v.parse::<usize>().expect("NV_TIMED_EMITTED_TOKENS"),
        Err(_) => CERT_EMITTED_TOKENS_512_MATCHES_THE_CHAMPION_LADDER_TIMED_WINDOW,
    }
}

fn cert_walk_classes_from_env_comma_list_unset_runs_all_three_so_deep_rungs_can_scope_to_one(
) -> Option<Vec<String>> {
    match std::env::var("NV_Q38_CERT_CLASSES") {
        Ok(v) => {
            let list: Vec<String> = v
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            assert!(
                !list.is_empty(),
                "NV_Q38_CERT_CLASSES is set but names no class; unset it to run all three"
            );
            for c in &list {
                assert!(
                    matches!(c.as_str(), "agentic" | "tiled" | "freeform"),
                    "NV_Q38_CERT_CLASSES names unknown class {c}; the classes are \
                     agentic, tiled, freeform"
                );
            }
            Some(list)
        }
        Err(_) => None,
    }
}

fn snapshot_dir() -> std::path::PathBuf {
    match std::env::var("NV_QWEN38_DIR") {
        Ok(d) if !d.is_empty() => {
            let p = std::path::PathBuf::from(&d);
            assert!(p.is_dir(), "NV_QWEN38_DIR={d} is not a directory");
            p
        }
        _ => hub_dirs::snapshot(NVFP4_REPO, &["config.json", "tokenizer.json"]).unwrap_or_else(
            || {
                panic!(
                    "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no hydrated {NVFP4_REPO} \
                     snapshot; set NV_QWEN38_DIR"
                )
            },
        ),
    }
}

fn snapshot_rev_label(dir: &std::path::Path) -> String {
    dir.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn load_base(
    dir: &std::path::Path,
    device: &Device,
) -> nv_models::qwen3_5_moe::Qwen3Moe {
    use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = nv_weights::QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = nv_weights::WeightLoader::open_dir(dir, device).expect("weights");
    Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, device)
        .expect("track-2 dense NVFP4 loader")
}

fn cert_prompt_variants_from_env_default_1_variant0_reproduces_the_banked_prompts() -> usize {
    match std::env::var("NV_Q38_CERT_VARIANTS") {
        Ok(v) => {
            let n: usize = v
                .parse()
                .expect("NV_Q38_CERT_VARIANTS must be a variant count");
            assert!(
                (1..=4).contains(&n),
                "NV_Q38_CERT_VARIANTS must be 1..=4: the agentic rename table holds 4 \
                 disjoint pairs and the freeform corpus is sliced per variant"
            );
            n
        }
        Err(_) => 1,
    }
}

fn cert_variant_start_from_env_default_0_isolates_a_failing_variant_in_a_fresh_process() -> usize {
    match std::env::var("NV_Q38_CERT_VARIANT_START") {
        Ok(v) => {
            let n: usize = v
                .parse()
                .expect("NV_Q38_CERT_VARIANT_START must be a variant index");
            assert!(n < 4, "NV_Q38_CERT_VARIANT_START must be 0..=3");
            n
        }
        Err(_) => 0,
    }
}

fn mean_and_sample_sd(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let sd = if xs.len() < 2 {
        0.0
    } else {
        (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    };
    (mean, sd)
}

fn median_of(ms: &[f64]) -> f64 {
    let mut sorted = ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 once; set NV_QWEN38_SERVING_TEST=1 and \
            NV_DRAFTER=mtp -- acceptance-preservation half of the tctile quality cert: one \
            graphed MTP session per workload class (agentic/tiled/freeform) generates 512 \
            emitted tokens on whichever verify arm the env routes (NV_Q38_MK_VERIFY_TCTILE set \
            picks tctile, unset runs the shipped kvgate basis; ONE arm per process because \
            graph capture bakes the arm), then the emitted text is teacher-force re-scored \
            through the graphed plain t=1 decode path, which both arms share, so accept_rate, \
            tokens_per_round and self-scored NLL of the emitted text are comparable arm to arm"]
fn qwen38_tctile_cert_accept_and_self_scored_emitted_nll_three_classes_one_arm_per_process() {
    use nv_models::graph_engine::GraphedQwen3Moe;
    use nv_models::qwen3_5_moe::nv_q38_mk_verify_tctile_env_opt_in_one_launch_mma_qk_tiles_read_kv_once_for_all_m_by_group_rows;
    use nv_specdecode::qwen38_mtp::{
        mtp_chain_depth_from_env, mtp_draft_dir_override_from_env, mtp_drafter_selected_from_env,
        Qwen38DenseMtpHead, Qwen38MtpGraphedDecodeSession,
    };

    require_gate("NV_QWEN38_SERVING_TEST");
    assert!(
        mtp_drafter_selected_from_env(),
        "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_DRAFTER=mtp so the flags \
         carry the champion spec env"
    );
    let depth = cert_depth_from_env_single_value_default_8192();
    let want_emitted = cert_emitted_tokens_from_env_default_512();
    let arm = match nv_q38_mk_verify_tctile_env_opt_in_one_launch_mma_qk_tiles_read_kv_once_for_all_m_by_group_rows()
    {
        Some(min_total) => {
            assert!(
                min_total <= depth,
                "tctile min_total {min_total} sits above depth {depth}: the route would run the \
                 mk ladder and the arm=tctile label would lie"
            );
            "tctile"
        }
        None => "kvgate-control",
    };
    let mut arm = arm.to_string();
    for (env, tag) in [
        ("NV_FP8_LT_NOWS_12288", "+nows12288"),
        ("NV_Q38_W4A8_MK", "+w4a8"),
        ("NV_Q38_DRAFT_VOCAB_PREFIX", "+vocabpfx"),
        ("NV_Q38_GDN_NVFP4_VERIFY", "+gdnnv4"),
        ("NV_Q38_ROPE_FUSED", "+ropef"),
        ("NV_Q38_ATTN_GATE_FUSED", "+gatef"),
    ] {
        if std::env::var(env).map(|v| !v.is_empty() && v != "0").unwrap_or(false) {
            arm.push_str(tag);
        }
    }
    let dir = snapshot_dir();
    let rev = snapshot_rev_label(&dir);
    let tok =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let base = load_base(&dir, &device);
    let max_seq = depth + CERT_KV_HEADROOM_1024_COVERS_ANCHOR_AND_ROUND_OVERSHOOT;
    let mut engine = GraphedQwen3Moe::new(base, &device, max_seq).expect("graphed engine");
    let mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &dir,
        engine.underlying(),
        &device,
    )
    .expect("MTP head loads from the shipped model_mtp.safetensors");
    let k = mtp_chain_depth_from_env();
    assert!(
        engine.cache().mk_verify_route_available(k + 1),
        "mk verify route unavailable for rows {}: acceptance here would measure the full-dequant \
         fallback, not the arm under certification",
        k + 1
    );

    let variants = cert_prompt_variants_from_env_default_1_variant0_reproduces_the_banked_prompts();
    let selected =
        cert_walk_classes_from_env_comma_list_unset_runs_all_three_so_deep_rungs_can_scope_to_one();
    let vstart = cert_variant_start_from_env_default_0_isolates_a_failing_variant_in_a_fresh_process();
    let mut per_class: std::collections::BTreeMap<&'static str, (Vec<f64>, Vec<f64>)> =
        std::collections::BTreeMap::new();
    for variant in vstart..vstart + variants {
    for (class, prompt) in sufspec_classes::three_class_prompt_variants(&tok, depth, variant) {
        if let Some(list) = &selected {
            if !list.iter().any(|c| c == class) {
                continue;
            }
        }
        let t_pre = Instant::now();
        let mut session = Qwen38MtpGraphedDecodeSession::start(&mut engine, &mtp, k, &prompt)
            .unwrap_or_else(|e| panic!("graphed mtp session start on class {class}: {e:#}"));
        let prefill_s = t_pre.elapsed().as_secs_f64();
        let mut emitted_stream: Vec<u32> = Vec::new();
        let mut round_ms: Vec<f64> = Vec::new();
        let t0 = Instant::now();
        while emitted_stream.len() < want_emitted {
            assert!(
                session.round_fits(),
                "kv headroom exhausted mid-generation at committed {}",
                session.committed_len()
            );
            let r0 = Instant::now();
            let r = session
                .round()
                .unwrap_or_else(|e| panic!("mtp round on class {class}: {e:#}"));
            round_ms.push(r0.elapsed().as_secs_f64() * 1e3);
            emitted_stream.extend(&r);
        }
        let wall_s = t0.elapsed().as_secs_f64();
        let (rounds, drafted, accepted) = (
            session.stats.rounds,
            session.stats.drafted,
            session.stats.accepted,
        );
        drop(session);
        assert!(
            rounds > 0 && drafted > 0 && accepted > 0,
            "degenerate run on class {class}: rounds={rounds} drafted={drafted} \
             accepted={accepted}; an accept_rate measured on nothing certifies nothing"
        );

        engine.reset().expect("reset before self-scoring");
        engine
            .install_grouped_moe()
            .expect("routing install before self-scoring");
        let mut fed = 0usize;
        let mut last_row: Vec<f32> = Vec::new();
        while fed < prompt.len() {
            let n = CERT_PREFILL_CHUNK_512_MATCHES_THE_TCTILE_GATE_FEED.min(prompt.len() - fed);
            last_row = engine
                .prefill(&prompt[fed..fed + n])
                .unwrap_or_else(|e| panic!("self-score prefill at {fed}: {e:#}"));
            fed += n;
        }
        let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
        acc.add_position_full_vocab_row(&last_row, emitted_stream[0]);
        for i in 0..emitted_stream.len() - 1 {
            let row = engine
                .forward_decode_logits_vec(emitted_stream[i])
                .unwrap_or_else(|e| panic!("self-score decode step {i}: {e:#}"));
            acc.add_position_full_vocab_row(&row, emitted_stream[i + 1]);
        }
        assert_eq!(
            acc.scored_positions,
            emitted_stream.len(),
            "self-scoring must score every emitted token exactly once"
        );

        let emitted = emitted_stream.len();
        let accept_rate = accepted as f64 / drafted as f64;
        let self_nll = acc.sum_neg_ln_p_f64 / acc.scored_positions as f64;
        per_class.entry(class).or_default().0.push(accept_rate);
        per_class.entry(class).or_default().1.push(self_nll);
        eprintln!(
            "CTX-SCALING qwen38-tctile-cert class={class} variant={variant} depth={depth} \
             arm={arm} k={k} \
             rounds={rounds} steps={emitted} warmup_steps=0 drafted={drafted} \
             accepted={accepted} accept_rate={accept_rate:.4} tokens_per_round={:.3} \
             median_ms_tok={:.3} tok_s={:.1} round_ms_median={:.1} prefill_s={prefill_s:.1} \
             self_nll={self_nll:.4} self_ppl={:.3} self_top1={:.4} emitted_fnv={:016x} \
             checkpoint={CHECKPOINT_LABEL}@{rev}",
            emitted as f64 / rounds as f64,
            1000.0 * wall_s / emitted as f64,
            emitted as f64 / wall_s,
            median_of(&round_ms),
            acc.perplexity_exp_of_mean_neg_ln_p(),
            acc.top1_accuracy(),
            sufspec_classes::fnv1a64_over_u32_le(&emitted_stream),
        );
        let vtag = if variant == 0 {
            String::new()
        } else {
            format!("-v{variant}")
        };
        eprintln!(
            "PPL qwen38-cert-emitted-{class}{vtag}-{arm} {CHECKPOINT_LABEL}@{rev} tokens={} \
             ppl={:.3} acc={:.4} basis=self_scored_emitted_text_graphed_plain_t1_decode",
            acc.scored_positions,
            acc.perplexity_exp_of_mean_neg_ln_p(),
            acc.top1_accuracy(),
        );
    }
    }
    for (class, (accepts, nlls)) in &per_class {
        let (a_mean, a_sd) = mean_and_sample_sd(accepts);
        let (n_mean, n_sd) = mean_and_sample_sd(nlls);
        eprintln!(
            "CERT-VARIANCE qwen38-tctile-cert class={class} depth={depth} arm={arm} \
             n={} accept_mean={a_mean:.4} accept_sd={a_sd:.4} self_nll_mean={n_mean:.4} \
             self_nll_sd={n_sd:.4} basis=prompt_variants_within_class_greedy_t1",
            accepts.len(),
        );
    }
}

struct VerifyRowTopScores {
    ids: [u32; TOPK_8_SCORES_KEPT_PER_VERIFY_ROW_SO_A_FLIPPED_TOKENS_SCORE_IN_THE_OTHER_ARM_IS_ON_HAND],
    vals: [f32; TOPK_8_SCORES_KEPT_PER_VERIFY_ROW_SO_A_FLIPPED_TOKENS_SCORE_IN_THE_OTHER_ARM_IS_ON_HAND],
}

fn top8_of_row(row: &[f32]) -> VerifyRowTopScores {
    let k = TOPK_8_SCORES_KEPT_PER_VERIFY_ROW_SO_A_FLIPPED_TOKENS_SCORE_IN_THE_OTHER_ARM_IS_ON_HAND;
    let mut ids = [0u32; 8];
    let mut vals = [f32::NEG_INFINITY; 8];
    for (i, &v) in row.iter().enumerate() {
        assert!(v.is_finite(), "non-finite verify logit at vocab index {i}");
        if v <= vals[k - 1] {
            continue;
        }
        let mut j = k - 1;
        while j > 0 && v > vals[j - 1] {
            ids[j] = ids[j - 1];
            vals[j] = vals[j - 1];
            j -= 1;
        }
        ids[j] = i as u32;
        vals[j] = v;
    }
    VerifyRowTopScores { ids, vals }
}

fn contested_gap_rel_top1_minus_other_or_eighth_as_a_floor_when_the_other_left_the_top8(
    rec: &VerifyRowTopScores,
    other_top1: u32,
) -> f32 {
    let denom = rec.vals[0].abs().max(1.0);
    let other_val = rec
        .ids
        .iter()
        .position(|&i| i == other_top1)
        .map(|j| rec.vals[j])
        .unwrap_or(rec.vals[7]);
    (rec.vals[0] - other_val) / denom
}

fn teacher_forced_verify_walk_scoring_every_row_through_the_m_row_verify_attention(
    base: &nv_models::qwen3_5_moe::Qwen3Moe,
    device: &Device,
    stream: &[u32],
    prefill_len: usize,
    m: usize,
) -> (Vec<VerifyRowTopScores>, TeacherForcedNllFp32SoftmaxF64Sum) {
    let mut cache = base
        .new_kv_cache(stream.len() + m + 8)
        .expect("walk kv cache");
    assert!(
        cache.mk_verify_route_available(m),
        "mk verify route unavailable for rows {m}: the walk would score the full-dequant \
         fallback and both arms would trivially agree"
    );
    let mut fed = 0usize;
    while fed < prefill_len {
        let n = CERT_PREFILL_CHUNK_512_MATCHES_THE_TCTILE_GATE_FEED.min(prefill_len - fed);
        let tokens = Tensor::from_vec(stream[fed..fed + n].to_vec(), (1usize, n), device)
            .expect("prefill chunk");
        let pos = Tensor::from_vec(
            (fed as i32..(fed + n) as i32).collect::<Vec<i32>>(),
            n,
            device,
        )
        .expect("positions");
        base.forward_with_cache_dispatched_hidden_rows(&tokens, &pos, &mut cache, None, Some(1))
            .unwrap_or_else(|e| panic!("walk prefill at {fed}: {e:#}"));
        fed += n;
    }
    assert_eq!(
        cache.current_len(),
        prefill_len,
        "walk prefill must commit exactly the prompt"
    );
    let mut recs: Vec<VerifyRowTopScores> = Vec::new();
    let mut acc = TeacherForcedNllFp32SoftmaxF64Sum::new();
    let mut l = prefill_len;
    while l + m + 1 <= stream.len() {
        let batch = stream[l..l + m].to_vec();
        let bt = Tensor::from_vec(batch, (1usize, m), device).expect("verify batch");
        let bp = Tensor::from_vec(
            (l as i32..(l + m) as i32).collect::<Vec<i32>>(),
            m,
            device,
        )
        .expect("verify positions");
        cache.set_capture_lin_ckpts(true);
        let (vlogits, _hidden) = base
            .forward_with_cache_dispatched_hidden(&bt, &bp, &mut cache, None)
            .unwrap_or_else(|e| panic!("verify walk forward at {l}: {e:#}"));
        let flat: Vec<f32> = vlogits
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("host");
        assert_eq!(flat.len() % m, 0, "verify logit rows not divisible by m");
        let vocab = flat.len() / m;
        for r in 0..m {
            let row = &flat[r * vocab..(r + 1) * vocab];
            acc.add_position_full_vocab_row(row, stream[l + r + 1]);
            recs.push(top8_of_row(row));
        }
        cache.set_current_len(l + m);
        cache
            .rollback_lin_to(m)
            .unwrap_or_else(|e| panic!("walk lin commit at {l}: {e:#}"));
        l += m;
    }
    (recs, acc)
}

struct PairwiseDivergence {
    divergences: usize,
    round0_divergences: usize,
    sev_p50: f32,
    sev_p90: f32,
    sev_max: f32,
}

fn pairwise_divergence_router_amplified_so_severity_describes_it_and_never_gates(
    reference: &[VerifyRowTopScores],
    candidate: &[VerifyRowTopScores],
    m: usize,
) -> PairwiseDivergence {
    let mut severities: Vec<f32> = Vec::new();
    let mut round0 = 0usize;
    for (p, (a, b)) in reference.iter().zip(candidate.iter()).enumerate() {
        if a.ids[0] == b.ids[0] {
            continue;
        }
        let sev =
            contested_gap_rel_top1_minus_other_or_eighth_as_a_floor_when_the_other_left_the_top8(
                a,
                b.ids[0],
            )
            .max(
                contested_gap_rel_top1_minus_other_or_eighth_as_a_floor_when_the_other_left_the_top8(
                    b,
                    a.ids[0],
                ),
            );
        if p < m {
            round0 += 1;
        }
        severities.push(sev);
    }
    severities.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = severities.len();
    let pct = |q: f64| -> f32 {
        if n == 0 {
            0.0
        } else {
            severities[((q * (n - 1) as f64).round() as usize).min(n - 1)]
        }
    };
    PairwiseDivergence {
        divergences: n,
        round0_divergences: round0,
        sev_p50: pct(0.5),
        sev_p90: pct(0.9),
        sev_max: pct(1.0),
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 once; set NV_QWEN38_SERVING_TEST=1 and leave \
            NV_Q38_MK_VERIFY_KVSHARE and NV_Q38_MK_VERIFY_TCTILE unset -- distribution half of \
            the tctile quality cert: per workload class the same forced token stream (depth \
            prompt plus 2048 continuation of the class text) is teacher-forced through m=4 \
            verify forwards on the eager cuda arm three times, once per verify arm (mk, \
            kvshare, tctile; the arm envs are re-read per call so this suite drives them \
            itself and every walk sees the byte-identical committed stream), recording \
            verify-path NLL, top-1 accuracy and pairwise argmax divergences vs the mk arm; \
            divergences here are confidently different logits even for tie-class attention \
            deltas because 48 MoE routers amplify a 1-ulp score shift, per \
            STREAM_IDENTITY_IS_NOT_GATEABLE_FOR_SCORE_REORDERING_ARMS in qwen38_tctile_gate, \
            so the certification gates are aggregate: per-class and cross-class NLL deltas \
            and top-1 accuracy deltas on matched text, and a class gates the candidate only \
            when the shipped kvshare arm, walked as the accepted-arm baseline, itself sits \
            inside the bands there -- a class where the accepted arm exceeds them is a \
            high-variance router-tie regime (the agentic transcript at some depths) whose \
            single-trajectory aggregate swings both directions by whole tenths of a nat and \
            certifies nothing either way"]
fn qwen38_tctile_cert_teacher_forced_verify_walk_divergence_and_nll_mk_kvshare_tctile_matched_streams()
{
    use nv_models::qwen3_5_moe::MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING;
    const KVSHARE_ENV: &str = "NV_Q38_MK_VERIFY_KVSHARE";

    require_gate("NV_QWEN38_SERVING_TEST");
    for env in [
        KVSHARE_ENV,
        MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
        "NV_Q38_MK_VERIFY_ROWSHARED",
    ] {
        assert!(
            std::env::var(env).is_err(),
            "unset {env}: this suite drives every verify arm itself and a pre-set env would \
             silently walk an arm against itself"
        );
    }
    let depth = cert_depth_from_env_single_value_default_8192();
    let walk = CERT_WALK_CONTINUATION_TOKENS_2048_GIVES_511_M4_VERIFY_ROUNDS_PER_ARM;
    let m = CERT_WALK_ROWS_M4_THE_CHAIN_SHAPE_THE_K3_ROUTE_SERVES;
    let dir = snapshot_dir();
    let rev = snapshot_rev_label(&dir);
    let tok =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let selected = cert_walk_classes_from_env_comma_list_unset_runs_all_three_so_deep_rungs_can_scope_to_one();
    let class_streams: Vec<(&str, Vec<u32>)> =
        sufspec_classes::three_class_prompts(&tok, depth + walk)
            .into_iter()
            .filter(|(class, _)| {
                selected
                    .as_ref()
                    .map_or(true, |l| l.iter().any(|c| c == class))
            })
            .collect();
    assert!(
        !class_streams.is_empty(),
        "NV_Q38_CERT_CLASSES selected no class this suite builds"
    );
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let base = load_base(&dir, &device);

    let arms: [(&str, Option<&str>, Option<&str>); 3] = [
        ("mk", None, None),
        ("kvshare", Some("1"), None),
        ("tctile", None, Some("1")),
    ];
    let mut failures: Vec<String> = Vec::new();
    let mut tctile_deltas: Vec<f64> = Vec::new();
    for (class, stream) in class_streams {
        let mut walked: Vec<(&str, Vec<VerifyRowTopScores>, TeacherForcedNllFp32SoftmaxF64Sum)> =
            Vec::new();
        for (label, kvshare, tctile) in arms {
            match kvshare {
                Some(v) => std::env::set_var(KVSHARE_ENV, v),
                None => std::env::remove_var(KVSHARE_ENV),
            }
            match tctile {
                Some(v) => std::env::set_var(
                    MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
                    v,
                ),
                None => std::env::remove_var(
                    MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
                ),
            }
            let (recs, acc) =
                teacher_forced_verify_walk_scoring_every_row_through_the_m_row_verify_attention(
                    &base, &device, &stream, depth, m,
                );
            walked.push((label, recs, acc));
        }
        std::env::remove_var(KVSHARE_ENV);
        std::env::remove_var(
            MK_VERIFY_TCTILE_ENV_1_ALWAYS_GATE_AT_THE_KNEE_0_OR_UNSET_FOR_THE_SHIPPED_ROUTING,
        );
        let scored = walked[0].2.scored_positions;
        assert!(
            scored >= walk / 2,
            "class {class}: only {scored} verify positions scored; the walk is broken"
        );
        for (label, recs, acc) in walked.iter() {
            assert_eq!(
                acc.scored_positions, scored,
                "class {class} arm {label}: scored position count differs from the mk arm"
            );
            assert_eq!(
                recs.len(),
                scored,
                "class {class} arm {label}: record count differs from scored positions"
            );
        }
        let (_, mk_recs, mk_acc) = &walked[0];
        let mk_nll = mk_acc.sum_neg_ln_p_f64 / scored as f64;
        eprintln!(
            "PPL qwen38-vwalk-{class}-mk {CHECKPOINT_LABEL}@{rev} tokens={scored} ppl={:.4} \
             acc={:.4} depth={depth} m={m} basis=teacher_forced_m_row_verify_forward",
            mk_acc.perplexity_exp_of_mean_neg_ln_p(),
            mk_acc.top1_accuracy(),
        );
        let nll_band = NLL_DELTA_ABS_FLOOR_0_02_NATS_PLUS_2_PCT_RELATIVE_BECAUSE_ROUTER_TIE_FLIPS_MOVE_SINGLE_POSITIONS_BY_WHOLE_NATS_WITHOUT_SYSTEMATIC_DAMAGE
            + NLL_DELTA_REL_2_PCT_OF_THE_REFERENCE_ARMS_MEAN_NLL * mk_nll;
        let mut class_gates_the_candidate_because_the_accepted_arm_sits_inside_the_bands = true;
        for (label, recs, acc) in walked.iter().skip(1) {
            let d = pairwise_divergence_router_amplified_so_severity_describes_it_and_never_gates(
                mk_recs, recs, m,
            );
            let nll = acc.sum_neg_ln_p_f64 / scored as f64;
            let nll_delta = nll - mk_nll;
            let acc_delta = acc.top1_accuracy() - mk_acc.top1_accuracy();
            let inside = nll_delta.abs() <= nll_band
                && acc_delta.abs() <= TOP1_ACC_DELTA_BAND_1_PCT_ON_MATCHED_FORCED_TEXT;
            if *label == "kvshare" {
                class_gates_the_candidate_because_the_accepted_arm_sits_inside_the_bands = inside;
            }
            eprintln!(
                "PPL qwen38-vwalk-{class}-{label} {CHECKPOINT_LABEL}@{rev} tokens={scored} \
                 ppl={:.4} acc={:.4} depth={depth} m={m} \
                 basis=teacher_forced_m_row_verify_forward ref_arm=mk \
                 divergences={} divergences_per_1000={:.2} round0_divergences={} \
                 div_sev_p50={:.5} div_sev_p90={:.5} div_sev_max={:.5} \
                 nll_ref={mk_nll:.5} nll_delta={nll_delta:+.5} acc_delta={acc_delta:+.4} \
                 nll_band={nll_band:.5} class_gated={}",
                acc.perplexity_exp_of_mean_neg_ln_p(),
                acc.top1_accuracy(),
                d.divergences,
                d.divergences as f64 * 1000.0 / scored as f64,
                d.round0_divergences,
                d.sev_p50,
                d.sev_p90,
                d.sev_max,
                class_gates_the_candidate_because_the_accepted_arm_sits_inside_the_bands as u8,
            );
            if *label == "tctile"
                && class_gates_the_candidate_because_the_accepted_arm_sits_inside_the_bands
            {
                if !inside {
                    failures.push(format!(
                        "class {class}: tctile verify-path nll delta {nll_delta:+.5} (band \
                         {nll_band:.5}) or top-1 delta {acc_delta:+.4} (band \
                         {TOP1_ACC_DELTA_BAND_1_PCT_ON_MATCHED_FORCED_TEXT}) vs mk exceeds \
                         what the accepted kvshare arm stays inside on this class"
                    ));
                }
                tctile_deltas.push(nll_delta);
            }
        }
    }
    assert!(
        !tctile_deltas.is_empty(),
        "no class gated the candidate: every class put the accepted kvshare arm outside the \
         bands, so this depth measured only high-variance regimes and certifies nothing"
    );
    let mean_tctile_delta = tctile_deltas.iter().sum::<f64>() / tctile_deltas.len() as f64;
    eprintln!(
        "TCTILE-CERT-VWALK-SUMMARY gated_classes={} \
         mean_nll_delta_tctile_vs_mk_over_gated={mean_tctile_delta:+.5}",
        tctile_deltas.len()
    );
    if mean_tctile_delta
        > NLL_DELTA_CROSS_CLASS_MEAN_BAND_0_02_NATS_A_ONE_SIDED_SHIFT_ON_EVERY_CLASS_IS_DAMAGE_NOT_TIE_NOISE
    {
        failures.push(format!(
            "tctile mean verify-path nll delta {mean_tctile_delta:+.5} over the gated classes \
             exceeds the systematic-damage band, which is one-sided because damage is the \
             worse-nll direction and a mean improvement cannot fail a damage gate"
        ));
    }
    assert!(
        failures.is_empty(),
        "quality-level certification failed on matched forced streams:\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "loads the 27B checkpoint to answer one question: does the freeform-v1 8192 prompt \
            NaN the grouped-moe prefill WITHOUT any prior mtp session; set NV_QWEN38_SERVING_TEST=1"]
fn freeform_v1_prefill_last_row_finite_without_any_prior_session() {
    use nv_models::graph_engine::GraphedQwen3Moe;

    require_gate("NV_QWEN38_SERVING_TEST");
    let depth = cert_depth_from_env_single_value_default_8192();
    let dir = snapshot_dir();
    let tok =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let base = load_base(&dir, &device);
    let mut engine = GraphedQwen3Moe::new(base, &device, depth + 1024).expect("graphed engine");
    engine
        .install_grouped_moe()
        .expect("routing install before prefill");
    let prompt = sufspec_classes::three_class_prompt_variants(&tok, depth, 1)
        .into_iter()
        .find(|(class, _)| *class == "freeform")
        .expect("freeform variant 1 prompt")
        .1;
    let chunk: usize = std::env::var("NV_MIN_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CERT_PREFILL_CHUNK_512_MATCHES_THE_TCTILE_GATE_FEED);
    let start: usize = std::env::var("NV_MIN_START")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let prompt = &prompt[start..];
    let mut fed = 0usize;
    let mut last_row: Vec<f32> = Vec::new();
    while fed < prompt.len() {
        let n = chunk.min(prompt.len() - fed);
        last_row = engine
            .prefill(&prompt[fed..fed + n])
            .unwrap_or_else(|e| panic!("prefill at {fed}: {e:#}"));
        let bad = last_row.iter().filter(|v| !v.is_finite()).count();
        eprintln!(
            "[freeform-v1-min] prefill start={start} fed={} chunk={n} last_row_nonfinite={bad}",
            fed + n
        );
        fed += n;
    }
    let bad_final = last_row.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        bad_final, 0,
        "freeform-v1 prefill last row carries {bad_final} non-finite logits with no prior \
         session: the NaN is prefill-intrinsic, not session-state carryover"
    );
    let row = engine
        .forward_decode_logits_vec(prompt[prompt.len() - 1])
        .expect("one m=1 decode step");
    let bad_decode = row.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        bad_decode, 0,
        "first m=1 decode row carries {bad_decode} non-finite logits with no prior session"
    );
    eprintln!("[freeform-v1-min] finite through prefill and one decode step on the current \
               default scan path");
}
