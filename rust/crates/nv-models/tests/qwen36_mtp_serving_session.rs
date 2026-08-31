#![cfg(feature = "cuda")]

use candle_core::Device;
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{GroupedMoeDispatch, MoeDispatch, Qwen3Moe, Qwen3MoeConfig};
use nv_models::qwen3_5_mtp::{MtpHead, MtpMoeEagerDecodeSession, MtpMoeGraphedDecodeSession};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const MAX_NEW: usize = 64;
const MAX_SEQ: usize = 512;
const JUNK_DRAFT_TOKEN_ANY_POLICY_MUST_EMIT_THE_SAME_STREAM_IF_ROLLBACK_IS_LOSSLESS: u32 = 0;
const ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_MATCHING_THE_QWEN38_CROSS_SHAPE_TIE_CLASS_BAR:
    f32 = 1.0;

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

struct RunOut {
    generated: Vec<u32>,
    accept_rate: f64,
    tokens_per_round: f64,
    rounds: usize,
    emitted_argmax_margins_after_anchor: Vec<f32>,
    submitted_draft_len_hist: std::collections::BTreeMap<usize, usize>,
}

fn drive<S, R>(mut session: S, anchor: u32, stop: &[u32], mut round: R) -> (Vec<u32>, S)
where
    R: FnMut(&mut S) -> anyhow::Result<Vec<u32>>,
{
    let mut generated: Vec<u32> = vec![anchor];
    if stop.contains(&anchor) {
        return (generated, session);
    }
    'rounds: while generated.len() < MAX_NEW {
        let emitted = round(&mut session).expect("round");
        for &t in &emitted {
            generated.push(t);
            if stop.contains(&t) {
                break 'rounds;
            }
            if generated.len() >= MAX_NEW {
                break 'rounds;
            }
        }
    }
    (generated, session)
}

fn eager_run(
    base: &Qwen3Moe,
    mtp: &MtpHead,
    k: usize,
    disp: Option<&dyn MoeDispatch>,
    prompt: &[u32],
    stop: &[u32],
    junk: bool,
) -> RunOut {
    let session =
        MtpMoeEagerDecodeSession::start(base, mtp, k, prompt, MAX_SEQ, disp).expect("eager start");
    let anchor = session.anchor();
    let junk_drafts =
        vec![JUNK_DRAFT_TOKEN_ANY_POLICY_MUST_EMIT_THE_SAME_STREAM_IF_ROLLBACK_IS_LOSSLESS; k];
    let (generated, session) = drive(session, anchor, stop, |s| {
        if junk {
            s.round_with_drafts_from_a_clairvoyant_test_oracle(&junk_drafts)
        } else {
            s.round()
        }
    });
    RunOut {
        generated,
        accept_rate: session.stats.accept_rate(),
        tokens_per_round: session.stats.tokens_per_round(),
        rounds: session.stats.rounds,
        emitted_argmax_margins_after_anchor: session
            .emitted_argmax_margins_after_anchor_recorded_only_under_p_min_so_the_default_arm_skips_the_full_logits_readback
            .clone(),
        submitted_draft_len_hist: session.submitted_draft_len_hist.clone(),
    }
}

fn graphed_run(
    engine: &mut GraphedQwen3Moe,
    mtp: &MtpHead,
    k: usize,
    prompt: &[u32],
    stop: &[u32],
    junk: bool,
) -> RunOut {
    let session = MtpMoeGraphedDecodeSession::start(engine, mtp, k, prompt).expect("graphed start");
    let anchor = session.anchor();
    let junk_drafts =
        vec![JUNK_DRAFT_TOKEN_ANY_POLICY_MUST_EMIT_THE_SAME_STREAM_IF_ROLLBACK_IS_LOSSLESS; k];
    let (generated, session) = drive(session, anchor, stop, |s| {
        if junk {
            s.round_with_drafts_from_a_clairvoyant_test_oracle(&junk_drafts)
        } else {
            s.round()
        }
    });
    RunOut {
        generated,
        accept_rate: session.stats.accept_rate(),
        tokens_per_round: session.stats.tokens_per_round(),
        rounds: session.stats.rounds,
        emitted_argmax_margins_after_anchor: session
            .emitted_argmax_margins_aligned_to_tokens_after_the_anchor_so_a_cross_shape_divergence_can_be_triaged_tie_vs_corruption
            .clone(),
        submitted_draft_len_hist: session.submitted_draft_len_hist.clone(),
    }
}

fn divergence_is_absent_or_tie_class_and_needs_a_determinism_rerun(
    arm: &str,
    real: &RunOut,
    junk: &RunOut,
    p_min: Option<f32>,
    tok: &Tokenizer,
) -> bool {
    if p_min.is_none() {
        assert_eq!(
            real.generated, junk.generated,
            "{arm} arm: the emitted stream depends on the draft policy with the p_min cut OFF \
             -- the rollback or hidden reanchor is lossy"
        );
        return false;
    }
    if real.generated == junk.generated {
        return false;
    }
    let i = real
        .generated
        .iter()
        .zip(junk.generated.iter())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| real.generated.len().min(junk.generated.len()));
    assert!(
        i >= 1,
        "{arm} arm: adaptive arms disagree on the prefill anchor itself: the prefill is \
         shape-fixed, so this is state corruption, not a tie"
    );
    let real_margin = real
        .emitted_argmax_margins_after_anchor
        .get(i - 1)
        .copied()
        .unwrap_or_else(|| {
            panic!(
                "{arm} arm: divergence at idx {i} but the real arm recorded only {} margins",
                real.emitted_argmax_margins_after_anchor.len()
            )
        });
    let junk_margin = junk
        .emitted_argmax_margins_after_anchor
        .get(i - 1)
        .copied()
        .unwrap_or_else(|| {
            panic!(
                "{arm} arm: divergence at idx {i} but the junk arm recorded only {} margins",
                junk.emitted_argmax_margins_after_anchor.len()
            )
        });
    eprintln!(
        "[q36-mtp-serve] arm={arm} adaptive-tie-divergence idx={i} real_tok={:?} junk_tok={:?} \
         real_margin={real_margin:.4} junk_margin={junk_margin:.4} real_text={:?} junk_text={:?}",
        real.generated.get(i),
        junk.generated.get(i),
        tok.decode(&real.generated, true).unwrap_or_default(),
        tok.decode(&junk.generated, true).unwrap_or_default()
    );
    assert!(
        real_margin <= ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_MATCHING_THE_QWEN38_CROSS_SHAPE_TIE_CLASS_BAR
            || junk_margin
                <= ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_MATCHING_THE_QWEN38_CROSS_SHAPE_TIE_CLASS_BAR,
        "{arm} arm: adaptive divergence at idx {i} has BOTH arms confident (argmax margins \
         real={real_margin} junk={junk_margin}): a p_min cut may flip greedy only where at \
         least one trajectory sits inside the cross-shape argmax noise band; two confident \
         computations of the same committed prefix disagreeing is the \
         rollback/reanchor/lin-ckpt/graph-replay bug class",
    );
    true
}

#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B checkpoint; set NV_QWEN36_MTP_SERVE=1 -- acceptance \
            preservation for the serving sessions: on both the eager arm and the graphed \
            verify-lane arm, junk drafts must emit the token stream real drafts emit (every \
            emitted token comes from the same m=k+1 verify forward, so any divergence is a \
            rollback/reanchor bug, not kernel numerics), and the real-draft arm must report \
            nonzero acceptance"]
fn serving_sessions_emit_a_draft_policy_invariant_stream_and_report_acceptance() {
    if std::env::var("NV_QWEN36_MTP_SERVE").as_deref() != Ok("1") {
        panic!("set NV_QWEN36_MTP_SERVE=1 to run (it must never silently skip)");
    }
    let dir = snapshot_with_mtp_shard();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw).expect("config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);
    let mtp = MtpHead::from_safetensors(dir.join("model_mtp.safetensors"), &base, &device)
        .expect("the shipped MoE MTP shard must load");

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let q = std::env::var("NV_QWEN36_Q")
        .unwrap_or_else(|_| "What is the capital of France? Answer in one short sentence.".into());
    let prompt_text =
        format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let prompt: Vec<u32> = tok
        .encode(prompt_text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.token_to_id(s))
        .collect();
    let k = nv_models::qwen3_5_mtp::qwen_mtp_k();

    let p_min = nv_models::qwen3_5_mtp::q36_mtp_draft_p_min_from_env();
    let p_min_flag = p_min
        .map(|p| format!("{p:.2}"))
        .unwrap_or_else(|| "off".into());

    let dispatch = GroupedMoeDispatch::from_model(&base).expect("grouped dispatch");
    let disp: Option<&dyn MoeDispatch> = Some(&dispatch as &dyn MoeDispatch);
    let eager_real = eager_run(&base, &mtp, k, disp, &prompt, &stop, false);
    let eager_junk = eager_run(&base, &mtp, k, disp, &prompt, &stop, true);

    let eager_text = tok.decode(&eager_real.generated, false).unwrap_or_default();
    eprintln!(
        "[q36-mtp-serve] EAGER basis=(RedHatAI/Qwen3.6-35B-A3B-NVFP4 @ {}, k={k}, \
         p_min={p_min_flag}, greedy, max_new={MAX_NEW}, grouped dispatch): real rounds={} \
         accept={:.3} tok/round={:.2} submitted_hist={:?} junk rounds={} text={eager_text:?}",
        dir.file_name().unwrap_or_default().to_string_lossy(),
        eager_real.rounds,
        eager_real.accept_rate,
        eager_real.tokens_per_round,
        eager_real.submitted_draft_len_hist,
        eager_junk.rounds,
    );
    if divergence_is_absent_or_tie_class_and_needs_a_determinism_rerun(
        "eager",
        &eager_real,
        &eager_junk,
        p_min,
        &tok,
    ) {
        let eager_again = eager_run(&base, &mtp, k, disp, &prompt, &stop, false);
        assert_eq!(
            eager_real.generated, eager_again.generated,
            "eager arm: adaptive emitted stream must be deterministic across identical \
             sessions; nondeterminism here is state corruption, not a tie"
        );
    }
    drop(dispatch);
    assert!(
        eager_real.accept_rate > 0.0,
        "eager arm: zero acceptance means the MoE MTP head never drafted a token the base \
         agreed with"
    );
    assert!(
        eager_real.generated.iter().any(|t| !stop.contains(t)),
        "eager arm generated no content tokens; the invariance above is measured on nothing"
    );

    let mut engine = GraphedQwen3Moe::new(base, &device, MAX_SEQ).expect("graphed engine");
    engine.install_grouped_moe().expect("grouped install");
    let graphed_real = graphed_run(&mut engine, &mtp, k, &prompt, &stop, false);
    let graphed_junk = graphed_run(&mut engine, &mtp, k, &prompt, &stop, true);
    let graphed_text = tok
        .decode(&graphed_real.generated, false)
        .unwrap_or_default();
    eprintln!(
        "[q36-mtp-serve] GRAPHED-VERIFY basis=(same checkpoint, k={k}, p_min={p_min_flag}, \
         greedy, max_new={MAX_NEW}, verify lane rows<={} capture_active={}): real rounds={} \
         accept={:.3} tok/round={:.2} submitted_hist={:?} junk rounds={} text={graphed_text:?}",
        k + 1,
        engine.capture_active(),
        graphed_real.rounds,
        graphed_real.accept_rate,
        graphed_real.tokens_per_round,
        graphed_real.submitted_draft_len_hist,
        graphed_junk.rounds,
    );
    if divergence_is_absent_or_tie_class_and_needs_a_determinism_rerun(
        "graphed",
        &graphed_real,
        &graphed_junk,
        p_min,
        &tok,
    ) {
        let graphed_again = graphed_run(&mut engine, &mtp, k, &prompt, &stop, false);
        assert_eq!(
            graphed_real.generated, graphed_again.generated,
            "graphed arm: adaptive emitted stream must be deterministic across identical \
             sessions; nondeterminism here is graph-replay/state corruption, not a tie"
        );
    }
    assert!(
        graphed_real.accept_rate > 0.0,
        "graphed arm: zero acceptance through the verify lane while the eager arm accepts \
         means the lane's logits or hidden rows are wrong"
    );
    assert!(
        graphed_real.generated.iter().any(|t| !stop.contains(t)),
        "graphed arm generated no content tokens; the invariance above is measured on nothing"
    );
}
