#![cfg(feature = "cuda")]

mod hub_dirs;

use candle_core::Device;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_specdecode::qwen38_mtp::{
    draft_lmhead_argmax_fuse_route_eligible_because_probs_top3_and_forced_drafts_need_logits,
    fused_argmax_packed_key_whose_inverted_index_makes_atomic_max_prefer_the_lowest_index,
    mtp_chain_depth_from_env, mtp_draft_dir_override_from_env,
    ordered_u32_key_from_f32_monotone_under_unsigned_max_with_neg_zero_tying_pos_zero,
    Qwen38DenseMtpHead, Qwen38MtpGraphedDecodeSession, MTP_WEIGHTS_FILE_NAME,
    NV_Q38_DRAFT_FAST_ENV, NV_Q38_DRAFT_LMHEAD_ARGMAX_FUSE_ENV, NV_Q38_DRAFT_LMHEAD_NVFP4_ENV,
};
use nv_weights::{QuantizationConfig, WeightLoader};
use tokenizers::Tokenizer;

mod common;
use common::render_real_chat_template_no_thinking;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const AB_MAX_SEQ: usize = 2048;
const AB_TARGET_COMPARED_ROUNDS_300_IS_HUNDREDS_OF_CHAINS_PER_THE_FUSION_GATE: usize = 300;
const AB_MIN_COMPARED_ROUNDS_100_BELOW_WHICH_THE_GATE_MEASURED_TOO_LITTLE: usize = 100;

const AB_PROMPT_SAME_CHANGELOG_BRACKET_AS_THE_DRAFT_PROFILE_INSTRUMENT: &str =
    "Continue this changelog with four more entries in exactly the same format, for versions \
     1.2.5 through 1.2.8:\n\n## 1.2.1\n- Fixed a crash when the config file is missing.\n- \
     Improved startup time by caching the model index.\n\n## 1.2.2\n- Fixed a crash when the \
     audio device is unplugged.\n- Improved startup time by lazy-loading the tokenizer.\n\n## \
     1.2.3\n- Fixed a crash when the network is unreachable.\n- Improved startup time by \
     deferring the license check.\n\n## 1.2.4\n- Fixed a crash when the cache directory is \
     read-only.\n- Improved startup time by precompiling the shaders.";

#[test]
fn ordered_key_is_monotone_over_f32_including_negatives_and_zero_signs_tie() {
    let samples: [f32; 12] = [
        f32::NEG_INFINITY,
        -3.4e38,
        -2.5,
        -1.0,
        -1.0e-30,
        -f32::MIN_POSITIVE,
        f32::MIN_POSITIVE,
        1.0e-30,
        1.0,
        2.5,
        3.4e38,
        f32::INFINITY,
    ];
    for w in samples.windows(2) {
        let (a, b) = (w[0], w[1]);
        assert!(
            ordered_u32_key_from_f32_monotone_under_unsigned_max_with_neg_zero_tying_pos_zero(a)
                < ordered_u32_key_from_f32_monotone_under_unsigned_max_with_neg_zero_tying_pos_zero(
                    b
                ),
            "ordered key must be strictly monotone: ord({a}) >= ord({b})"
        );
    }
    assert_eq!(
        ordered_u32_key_from_f32_monotone_under_unsigned_max_with_neg_zero_tying_pos_zero(-0.0),
        ordered_u32_key_from_f32_monotone_under_unsigned_max_with_neg_zero_tying_pos_zero(0.0),
        "-0.0 and +0.0 compare equal in float max, so their ordered keys must be equal or the \
         fused kernel would break a zero-sign tie by value instead of by index"
    );
}

#[test]
fn packed_key_under_max_prefers_the_higher_value_then_the_lower_index() {
    let lo = ordered_u32_key_from_f32_monotone_under_unsigned_max_with_neg_zero_tying_pos_zero(1.0);
    let hi = ordered_u32_key_from_f32_monotone_under_unsigned_max_with_neg_zero_tying_pos_zero(2.0);
    assert!(
        fused_argmax_packed_key_whose_inverted_index_makes_atomic_max_prefer_the_lowest_index(
            hi, 151935
        ) > fused_argmax_packed_key_whose_inverted_index_makes_atomic_max_prefer_the_lowest_index(
            lo, 0
        ),
        "a higher logit must beat a lower one at any index pair"
    );
    for (i, j) in [(0u32, 1u32), (0, 151935), (7, 8), (151934, 151935)] {
        assert!(
            fused_argmax_packed_key_whose_inverted_index_makes_atomic_max_prefer_the_lowest_index(
                hi, i
            ) > fused_argmax_packed_key_whose_inverted_index_makes_atomic_max_prefer_the_lowest_index(
                hi, j
            ),
            "equal logits: the packed key of index {i} must beat index {j} under max"
        );
    }
}

#[test]
fn fuse_route_falls_back_whenever_probs_top3_or_forced_drafts_need_materialized_logits() {
    let eligible =
        draft_lmhead_argmax_fuse_route_eligible_because_probs_top3_and_forced_drafts_need_logits;
    assert!(eligible(true, true, false, false, false));
    assert!(
        !eligible(true, true, true, false, false),
        "want_top_probs (the p_min cut's confidence read) needs sum-exp over logits the fused \
         kernel never computes; the route must fall back unfused"
    );
    assert!(!eligible(true, true, false, true, false));
    assert!(!eligible(true, true, false, false, true));
    assert!(
        !eligible(true, false, false, false, false),
        "without the nvfp4 twin the bf16 lm_head fallback stays on the unfused path"
    );
    assert!(!eligible(false, true, false, false, false));
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 plus the shipped MTP head; set \
            NV_QWEN38_SERVING_TEST=1 and the champion draft envs (the test pins \
            NV_Q38_DRAFT_LMHEAD_NVFP4=1 and NV_Q38_DRAFT_FAST=1 itself) -- at every committed \
            session state it drafts one k-chain with NV_Q38_DRAFT_LMHEAD_ARGMAX_FUSE off, then \
            on, then on again, over hundreds of rounds in one process, asserting the drafted \
            token streams bit-identical (same logits, same lowest-index argmax) and the fused \
            arm deterministic; also asserts the fuse env with top-probs requested still \
            returns real probs through the unfused fallback"]
fn qwen38_mtp_lmfuse_on_vs_off_drafted_token_streams_bit_identical_over_hundreds_of_rounds() {
    if std::env::var("NV_QWEN38_SERVING_TEST").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_QWEN38_SERVING_TEST=1");
    }
    std::env::set_var(NV_Q38_DRAFT_LMHEAD_NVFP4_ENV, "1");
    std::env::set_var(NV_Q38_DRAFT_FAST_ENV, "1");
    std::env::remove_var(NV_Q38_DRAFT_LMHEAD_ARGMAX_FUSE_ENV);
    let k = mtp_chain_depth_from_env();

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
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("track-2 dense NVFP4 loader must load Qwen3.8-27B before this gate can run");
    drop(weights);
    let mut engine = nv_models::graph_engine::GraphedQwen3Moe::new(base, &device, AB_MAX_SEQ)
        .expect("graphed engine over the dense arm");
    let mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &dir,
        engine.underlying(),
        &device,
    )
    .expect("MTP head loads from the shipped model_mtp.safetensors");
    assert!(
        mtp.draft_lm_head_nvfp4_twin_active(),
        "{NV_Q38_DRAFT_LMHEAD_NVFP4_ENV}=1 was set before load but no twin is resident; the \
         fused route would silently never engage and this gate would compare the unfused path \
         to itself"
    );

    let prompt: Vec<u32> = tok
        .encode(
            render_real_chat_template_no_thinking(
                &dir,
                AB_PROMPT_SAME_CHANGELOG_BRACKET_AS_THE_DRAFT_PROFILE_INSTRUMENT,
            )
            .as_str(),
            false,
        )
        .expect("encode")
        .get_ids()
        .to_vec();

    let mut session = Qwen38MtpGraphedDecodeSession::start(&mut engine, &mtp, k, &prompt)
        .expect("graphed mtp session start");
    let mut compared_rounds = 0usize;
    let mut fused_engaged_rounds = 0usize;
    while compared_rounds < AB_TARGET_COMPARED_ROUNDS_300_IS_HUNDREDS_OF_CHAINS_PER_THE_FUSION_GATE
        && session.round_fits()
    {
        std::env::remove_var(NV_Q38_DRAFT_LMHEAD_ARGMAX_FUSE_ENV);
        let (d_off, p_off) = session
            .drafts_and_top_probs_for_one_round_then_rewind_so_an_ab_gate_reads_the_chain_without_committing(
                false,
            )
            .expect("unfused draft chain probe");
        assert!(p_off.is_empty(), "probs were not requested on the off arm");
        std::env::set_var(NV_Q38_DRAFT_LMHEAD_ARGMAX_FUSE_ENV, "1");
        let (d_on, _) = session
            .drafts_and_top_probs_for_one_round_then_rewind_so_an_ab_gate_reads_the_chain_without_committing(
                false,
            )
            .expect("fused draft chain probe");
        let (d_on2, _) = session
            .drafts_and_top_probs_for_one_round_then_rewind_so_an_ab_gate_reads_the_chain_without_committing(
                false,
            )
            .expect("fused draft chain probe rerun");
        assert_eq!(
            d_on, d_on2,
            "round {compared_rounds}: the fused chain must be deterministic across identical \
             rewound states"
        );
        assert_eq!(
            d_off, d_on,
            "round {compared_rounds}: fused and unfused drafted token streams diverged; both \
             argmax the same bf16 logits with lowest-index ties, so this is a fused-kernel \
             defect, not a tie class"
        );
        let (d_probs, probs) = session
            .drafts_and_top_probs_for_one_round_then_rewind_so_an_ab_gate_reads_the_chain_without_committing(
                true,
            )
            .expect("fuse-env-on probs-requested probe");
        assert_eq!(
            probs.len(),
            k,
            "round {compared_rounds}: with the fuse env set AND top-probs requested the chain \
             must fall back to the unfused path and return one real prob per step, never \
             silently mis-serve"
        );
        assert!(
            probs.iter().all(|p| p.is_finite() && *p > 0.0 && *p <= 1.0),
            "round {compared_rounds}: fallback probs must be real softmax top probabilities, \
             got {probs:?}"
        );
        assert_eq!(
            d_probs, d_off,
            "round {compared_rounds}: the probs-requested fallback chain must draft the same \
             stream as the plain unfused chain"
        );
        fused_engaged_rounds += 1;
        compared_rounds += 1;
        std::env::remove_var(NV_Q38_DRAFT_LMHEAD_ARGMAX_FUSE_ENV);
        session.round().expect("committing round");
    }
    assert!(
        compared_rounds >= AB_MIN_COMPARED_ROUNDS_100_BELOW_WHICH_THE_GATE_MEASURED_TOO_LITTLE,
        "only {compared_rounds} rounds were compared; the session budget ended before the gate \
         reached hundreds of chains"
    );
    eprintln!(
        "[q38-mtp-lmfuse-gate] compared_rounds={compared_rounds} \
         fused_engaged_rounds={fused_engaged_rounds} k={k} prompt_toks={} \
         basis=(model=unsloth/Qwen3.8-27B-NVFP4 graphed-session doc-changelog-prompt)",
        prompt.len()
    );
}
