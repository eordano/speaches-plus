#![cfg(feature = "cuda")]

mod hub_dirs;

use candle_core::Device;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_specdecode::qwen38_mtp::{
    adaptive_verify_pins_the_env_unset_auto_dp4a_mlp_off_because_greedy_argmax_must_not_depend_on_which_m_the_cut_produced,
    mtp_chain_depth_from_env, mtp_draft_dir_override_from_env, mtp_draft_p_min,
    Qwen38DenseMtpHead, MTP_WEIGHTS_FILE_NAME, NV_Q38_MTP_P_MIN_ENV,
};
use nv_weights::{QuantizationConfig, WeightLoader};
use tokenizers::Tokenizer;

mod common;
use common::{
    generate_graphed, render_real_chat_template_no_thinking, GraphedRunRecord,
    ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_SHALLOW_FLIPS_MEASURE_0_0_TO_0_9_AND_DEEP_DRIFT_KEEPS_ONE_SIDE_UNDER_0_4,
};

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";

const ONOFF_MAX_SEQ: usize = 2048;
const ONOFF_MAX_NEW_96_LONG_ENOUGH_FOR_THE_CUT_TO_MEET_LOW_CONFIDENCE_SHORT_ENOUGH_FOR_A_GATE:
    usize = 96;
const ONOFF_MIN_EMITTED_9_SO_A_STOP_AT_THE_ANCHOR_CANNOT_PASS_VACUOUSLY: usize = 9;
const ONOFF_P_MIN_DEFAULT_0_75_IS_THE_MEASURED_ADAK_ARM: &str = "0.75";

const ONOFF_DOC_PROMPT_HIGH_CONFIDENCE_BRACKET_WHERE_THE_CUT_SHOULD_MOSTLY_NOT_FIRE: &str =
    "Continue this changelog with four more entries in exactly the same format, for versions \
     1.2.5 through 1.2.8:\n\n## 1.2.1\n- Fixed a crash when the config file is missing.\n- \
     Improved startup time by caching the model index.\n\n## 1.2.2\n- Fixed a crash when the \
     audio device is unplugged.\n- Improved startup time by lazy-loading the tokenizer.\n\n## \
     1.2.3\n- Fixed a crash when the network is unreachable.\n- Improved startup time by \
     deferring the license check.\n\n## 1.2.4\n- Fixed a crash when the cache directory is \
     read-only.\n- Improved startup time by precompiling the shaders.";

const ONOFF_CREATIVE_PROMPT_LOW_CONFIDENCE_BRACKET_GIVES_THE_CUT_ROOM_TO_FIRE: &str =
    "Write a six-line poem about a lighthouse keeper who befriends a storm.";

fn assert_on_off_pair_content_and_draft_work(
    off: &GraphedRunRecord,
    on: &GraphedRunRecord,
    k: usize,
    site: &str,
) {
    for (&len, &n) in &off.submitted_draft_len_hist {
        assert_eq!(
            len, k,
            "{site}: the p_min-off arm submitted a round of {len} drafts (x{n}) instead of the \
             fixed k={k}; the off arm IS today's default and must not vary its verify shape"
        );
    }
    let mut cut_fired = false;
    for (&len, &n) in &on.submitted_draft_len_hist {
        assert!(
            (2..=k).contains(&len),
            "{site}: the p_min-on arm submitted a round of {len} drafts (x{n}) outside [2, k={k}]; \
             the cut keeps at least two (the m>=3 graph-shape floor) and never exceeds NV_MTP_K"
        );
        if len < k {
            cut_fired = true;
        }
    }
    assert!(
        on.stats.drafted <= on.stats.rounds * k,
        "{site}: the p_min-on arm reports drafted={} over rounds={}, above the fixed-k budget \
         rounds*k={}; per-round submitted length must never exceed k",
        on.stats.drafted,
        on.stats.rounds,
        on.stats.rounds * k
    );
    if cut_fired {
        assert!(
            on.stats.drafted < on.stats.rounds * k,
            "{site}: the hist says the cut fired (some round submitted < k) but drafted={} == \
             rounds*k={}; the drafted stat counts SUBMITTED verify work and must drop when the \
             cut truncates",
            on.stats.drafted,
            on.stats.rounds * k
        );
    }
    assert!(
        !on.draft_top_prob_rounds.is_empty(),
        "{site}: the p_min-on arm recorded no drafter top-prob rounds; the cut cannot have \
         consulted confidence it never read"
    );
    assert!(
        off.ids.len() > ONOFF_MIN_EMITTED_9_SO_A_STOP_AT_THE_ANCHOR_CANNOT_PASS_VACUOUSLY
            && on.ids.len() > ONOFF_MIN_EMITTED_9_SO_A_STOP_AT_THE_ANCHOR_CANNOT_PASS_VACUOUSLY,
        "{site}: an arm emitted almost nothing (off={} on={}); a stub stream cannot gate content \
         identity",
        off.ids.len(),
        on.ids.len()
    );
    if !cut_fired {
        assert_eq!(
            off.ids, on.ids,
            "{site}: the cut never fired, so every p_min-on round verified the same m=k+1 shape \
             as the off arm and the streams must be bit-identical; a mismatch here is \
             rollback/reanchor/lin-ckpt/graph-replay corruption, not a tie"
        );
        return;
    }
    if off.ids == on.ids {
        return;
    }
    let i = off
        .ids
        .iter()
        .zip(on.ids.iter())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| off.ids.len().min(on.ids.len()));
    assert!(
        i >= 1,
        "{site}: on/off arms disagree on the prefill anchor itself; the prefill is shape-fixed, \
         so this is state corruption, not a tie"
    );
    let off_margin = off.emitted_argmax_margins_after_anchor[i - 1];
    let on_margin = on.emitted_argmax_margins_after_anchor[i - 1];
    eprintln!(
        "[q38-mtp-p-min-onoff] site={site} divergence idx={i} off_tok={} on_tok={} \
         off_margin={off_margin:.4} on_margin={on_margin:.4}",
        off.ids[i], on.ids[i]
    );
    assert!(
        off_margin
            <= ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_SHALLOW_FLIPS_MEASURE_0_0_TO_0_9_AND_DEEP_DRIFT_KEEPS_ONE_SIDE_UNDER_0_4
            || on_margin
                <= ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_SHALLOW_FLIPS_MEASURE_0_0_TO_0_9_AND_DEEP_DRIFT_KEEPS_ONE_SIDE_UNDER_0_4,
        "{site}: on/off streams diverge at idx={i} with BOTH arms confident (margins \
         off={off_margin} on={on_margin}); a p_min cut may flip greedy only inside the \
         cross-shape argmax noise band (rows-keyed verifies run shape-keyed kernel plans whose \
         ULP deltas compound through the recurrent lin state); two confident computations of the \
         same committed prefix disagreeing is the rollback/reanchor/lin-ckpt/graph-replay bug \
         class",
    );
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 plus the shipped MTP head; set \
            NV_QWEN38_SERVING_TEST=1 -- runs the SAME prompts through the graphed session with \
            NV_Q38_MTP_P_MIN unset (today's fixed-k default) and set (0.75 unless the env \
            carries another value), in one process on one engine, with the W4A8 mlp arm pinned \
            for BOTH arms so the comparison isolates the cut; gates the on/off content relation \
            (bit-identical when the cut never fires, tie-class divergence otherwise), the \
            per-round submitted-draft budget, and the on-arm's determinism; k from NV_MTP_K \
            (needs >= 2)"]
fn qwen38_mtp_p_min_cut_on_vs_off_same_prompt_stream_identity_or_tie_class_and_submitted_rows_never_exceed_fixed_k(
) {
    if std::env::var("NV_QWEN38_SERVING_TEST").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_QWEN38_SERVING_TEST=1");
    }
    let k = mtp_chain_depth_from_env();
    assert!(
        k >= 2,
        "the p_min cut needs k >= 2 (it keeps at least two drafts); set NV_MTP_K accordingly"
    );
    let caller_p_min = std::env::var(NV_Q38_MTP_P_MIN_ENV).ok();
    let p_min_raw = caller_p_min
        .clone()
        .unwrap_or_else(|| ONOFF_P_MIN_DEFAULT_0_75_IS_THE_MEASURED_ADAK_ARM.into());
    assert!(
        mtp_draft_p_min(Some(&p_min_raw)).is_some(),
        "{NV_Q38_MTP_P_MIN_ENV}={p_min_raw} does not parse as a threshold in (0,1); the on arm \
         would silently run fixed-k and this gate's label would lie"
    );

    let dir = hub_dirs::snapshot(
        NVFP4_REPO,
        &[
            "config.json",
            "tokenizer.json",
            "chat_template.jinja",
            MTP_WEIGHTS_FILE_NAME,
        ],
    )
    .expect("Qwen3.8-27B-NVFP4 snapshot with the MTP shard not found in the hub cache");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.token_to_id(s))
        .collect();
    assert!(!stop.is_empty(), "tokenizer lost the im_end/endoftext stop tokens");
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("track-2 dense NVFP4 loader must load Qwen3.8-27B before this gate can run");
    drop(weights);
    let mut engine = nv_models::graph_engine::GraphedQwen3Moe::new(base, &device, ONOFF_MAX_SEQ)
        .expect("graphed engine over the dense arm");
    let mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &dir,
        engine.underlying(),
        &device,
    )
    .expect("MTP head loads from the shipped model_mtp.safetensors");

    adaptive_verify_pins_the_env_unset_auto_dp4a_mlp_off_because_greedy_argmax_must_not_depend_on_which_m_the_cut_produced();

    let max_new = ONOFF_MAX_NEW_96_LONG_ENOUGH_FOR_THE_CUT_TO_MEET_LOW_CONFIDENCE_SHORT_ENOUGH_FOR_A_GATE;
    for (site, question) in [
        (
            "doc-changelog",
            ONOFF_DOC_PROMPT_HIGH_CONFIDENCE_BRACKET_WHERE_THE_CUT_SHOULD_MOSTLY_NOT_FIRE,
        ),
        (
            "creative-poem",
            ONOFF_CREATIVE_PROMPT_LOW_CONFIDENCE_BRACKET_GIVES_THE_CUT_ROOM_TO_FIRE,
        ),
    ] {
        let prompt: Vec<u32> = tok
            .encode(
                render_real_chat_template_no_thinking(&dir, question).as_str(),
                false,
            )
            .expect("encode")
            .get_ids()
            .to_vec();

        std::env::remove_var(NV_Q38_MTP_P_MIN_ENV);
        let off = generate_graphed(&mut engine, &mtp, k, &prompt, max_new, &stop, None);
        std::env::set_var(NV_Q38_MTP_P_MIN_ENV, &p_min_raw);
        let on = generate_graphed(&mut engine, &mtp, k, &prompt, max_new, &stop, None);
        let on_again = generate_graphed(&mut engine, &mtp, k, &prompt, max_new, &stop, None);
        assert_eq!(
            on.ids, on_again.ids,
            "{site}: the p_min-on emitted stream must be deterministic across identical \
             sessions; nondeterminism here is graph-replay/state corruption, not a tie"
        );

        let fmt_hist = |h: &std::collections::BTreeMap<usize, usize>| {
            h.iter()
                .map(|(l, n)| format!("{l}:{n}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        eprintln!(
            "[q38-mtp-p-min-onoff] site={site} k={k} p_min={p_min_raw} identical={} \
             off_rounds={} off_drafted={} off_hist={} on_rounds={} on_drafted={} on_hist={} \
             on_accept_hist={}",
            off.ids == on.ids,
            off.stats.rounds,
            off.stats.drafted,
            fmt_hist(&off.submitted_draft_len_hist),
            on.stats.rounds,
            on.stats.drafted,
            fmt_hist(&on.submitted_draft_len_hist),
            fmt_hist(&on.stats.accept_len_hist),
        );
        assert_on_off_pair_content_and_draft_work(&off, &on, k, site);
    }

    match caller_p_min {
        Some(v) => std::env::set_var(NV_Q38_MTP_P_MIN_ENV, v),
        None => std::env::remove_var(NV_Q38_MTP_P_MIN_ENV),
    }
}
