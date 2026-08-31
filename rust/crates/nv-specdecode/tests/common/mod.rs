#![allow(dead_code)]
#![allow(unused_imports)]

#[cfg(feature = "cuda")]
use minijinja::{context, Environment, Value as JinjaValue};
#[cfg(feature = "cuda")]
use std::path::Path;
#[cfg(feature = "wgpu")]
use nv_specdecode::wgpu_spec::SpecDims;
#[cfg(feature = "cuda")]
use nv_specdecode::qwen38_mtp::{Qwen38DenseMtpHead, Qwen38MtpGraphedDecodeSession};

#[cfg(feature = "cuda")]
pub fn raise_exception(msg: String) -> Result<JinjaValue, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

#[cfg(feature = "cuda")]
pub fn strftime_now(_fmt: String) -> Result<JinjaValue, minijinja::Error> {
    Ok(JinjaValue::from("1970-01-01"))
}

#[cfg(feature = "cuda")]
pub fn render_real_chat_template_no_thinking(dir: &Path, question: &str) -> String {
    let source = std::fs::read_to_string(dir.join("chat_template.jinja"))
        .expect("chat_template.jinja must ship with the checkpoint");
    let mut env = Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", raise_exception);
    env.add_function("strftime_now", strftime_now);
    env.add_template_owned("chat", source).expect("template compiles");
    env.get_template("chat")
        .unwrap()
        .render(context! {
            messages => vec![serde_json::json!({"role": "user", "content": question})],
            add_generation_prompt => true,
            enable_thinking => false,
        })
        .expect("the official template must render a plain user turn")
}

#[cfg(feature = "cuda")]
pub struct GraphedRunRecord {
    pub ids: Vec<u32>,
    pub stats: nv_models::qwen3_5_mtp::MtpSpecStats,
    pub round_emitted: Vec<usize>,
    pub round_wall_ms: Vec<f64>,
    pub submitted_draft_len_hist: std::collections::BTreeMap<usize, usize>,
    pub emitted_argmax_margins_after_anchor: Vec<f32>,
    pub draft_top_prob_rounds: Vec<Vec<f32>>,
}

#[cfg(feature = "cuda")]
pub fn generate_graphed(
    engine: &mut nv_models::graph_engine::GraphedQwen3Moe,
    mtp: &Qwen38DenseMtpHead,
    k: usize,
    prompt: &[u32],
    max_new: usize,
    stop: &[u32],
    junk_drafts: Option<u32>,
) -> GraphedRunRecord {
    let mut session = Qwen38MtpGraphedDecodeSession::start(engine, mtp, k, prompt)
        .expect("graphed mtp session start");
    let junk: Option<Vec<u32>> = junk_drafts.map(|t| vec![t; k]);
    let anchor = session.anchor();
    let mut ids: Vec<u32> = vec![anchor];
    let mut round_emitted: Vec<usize> = Vec::new();
    let mut round_wall_ms: Vec<f64> = Vec::new();
    if stop.contains(&anchor) {
        return GraphedRunRecord {
            ids,
            submitted_draft_len_hist: session.submitted_draft_len_hist,
            emitted_argmax_margins_after_anchor: session
                .emitted_argmax_margins_aligned_to_tokens_after_the_anchor_so_a_cross_shape_divergence_can_be_triaged_tie_vs_corruption,
            draft_top_prob_rounds: session
                .draft_top_prob_rounds_present_only_when_the_p_min_cut_or_round_debug_asked_the_chain_for_probs,
            stats: session.stats,
            round_emitted,
            round_wall_ms,
        };
    }
    'rounds: while ids.len() < max_new && session.round_fits() {
        let t0 = std::time::Instant::now();
        let emitted = match &junk {
            Some(j) => session
                .round_with_drafts_from_a_clairvoyant_test_oracle(j)
                .expect("graphed junk round"),
            None => session.round().expect("graphed round"),
        };
        round_wall_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        let mut taken = 0usize;
        for &t in emitted.iter() {
            ids.push(t);
            taken += 1;
            if stop.contains(&t) {
                round_emitted.push(taken);
                break 'rounds;
            }
            if ids.len() >= max_new {
                round_emitted.push(taken);
                break 'rounds;
            }
        }
        round_emitted.push(taken);
    }
    GraphedRunRecord {
        ids,
        submitted_draft_len_hist: session.submitted_draft_len_hist,
        emitted_argmax_margins_after_anchor: session
            .emitted_argmax_margins_aligned_to_tokens_after_the_anchor_so_a_cross_shape_divergence_can_be_triaged_tie_vs_corruption,
        draft_top_prob_rounds: session
            .draft_top_prob_rounds_present_only_when_the_p_min_cut_or_round_debug_asked_the_chain_for_probs,
        stats: session.stats,
        round_emitted,
        round_wall_ms,
    }
}

#[cfg(feature = "cuda")]
pub const ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_SHALLOW_FLIPS_MEASURE_0_0_TO_0_9_AND_DEEP_DRIFT_KEEPS_ONE_SIDE_UNDER_0_4:
    f32 = 1.0;

#[cfg(feature = "cuda")]
pub fn assert_draft_policy_invariance_exact_when_fixed_k_or_tie_class_when_the_p_min_cut_varies_the_verify_shape(
    engine: &mut nv_models::graph_engine::GraphedQwen3Moe,
    mtp: &Qwen38DenseMtpHead,
    k: usize,
    prompt: &[u32],
    max_new: usize,
    stop: &[u32],
    tok: &tokenizers::Tokenizer,
    spec: &GraphedRunRecord,
    junk: &GraphedRunRecord,
    site: &str,
) {
    if nv_specdecode::qwen38_mtp::mtp_draft_p_min_from_env().is_none() {
        assert_eq!(
            spec.ids, junk.ids,
            "draft-policy invariance broken at {site} through the GRAPHED verify: every emitted \
             token is the batched verify's own greedy continuation of the committed prefix, and \
             with the p_min cut off both arms verify the one m=k+1 shape, so the streams must be \
             identical; a mismatch is a rollback/reanchor/lin-ckpt/graph-replay bug, not numerics"
        );
        return;
    }
    if spec.ids == junk.ids {
        return;
    }
    let i = spec
        .ids
        .iter()
        .zip(junk.ids.iter())
        .position(|(a, b)| a != b)
        .expect("both arms cap at max_new, so differing streams differ at an index");
    assert!(
        i >= 1,
        "adaptive arms disagree on the prefill anchor itself at {site}: the prefill is \
         shape-fixed, so this is state corruption, not a tie"
    );
    let spec_margin = spec.emitted_argmax_margins_after_anchor[i - 1];
    let junk_margin = junk.emitted_argmax_margins_after_anchor[i - 1];
    eprintln!(
        "[q38-mtp-adaptive-tie-divergence] site={site} idx={i} spec_tok={} junk_tok={} \
         spec_margin={spec_margin:.4} junk_margin={junk_margin:.4} spec_ids={:?} junk_ids={:?} \
         spec_text={:?} junk_text={:?}",
        spec.ids[i],
        junk.ids[i],
        spec.ids,
        junk.ids,
        tok.decode(&spec.ids, true).unwrap_or_default(),
        tok.decode(&junk.ids, true).unwrap_or_default()
    );
    assert!(
        spec_margin
            <= ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_SHALLOW_FLIPS_MEASURE_0_0_TO_0_9_AND_DEEP_DRIFT_KEEPS_ONE_SIDE_UNDER_0_4
            || junk_margin
                <= ADAPTIVE_TIE_MARGIN_MAX_1_BF16_LOGIT_UNIT_SHALLOW_FLIPS_MEASURE_0_0_TO_0_9_AND_DEEP_DRIFT_KEEPS_ONE_SIDE_UNDER_0_4,
        "adaptive divergence at {site} idx={i} (spec {} vs junk {}) has BOTH arms confident: \
         argmax margins spec={spec_margin} junk={junk_margin}: a p_min cut may flip greedy where \
         at least one trajectory sits inside the cross-shape argmax noise band (rows-keyed \
         verifies run shape-keyed kernel plans whose ULP deltas compound through the recurrent \
         lin state, and the invariance-green fixed k=5 and k=7 streams already diverge at such a \
         tie); two confident computations of the same committed prefix disagreeing is the \
         rollback/reanchor/lin-ckpt/graph-replay bug class",
        spec.ids[i],
        junk.ids[i]
    );
    let spec_again = generate_graphed(engine, mtp, k, prompt, max_new, stop, None);
    assert_eq!(
        spec.ids, spec_again.ids,
        "adaptive emitted stream must be deterministic across identical sessions at {site}; \
         nondeterminism here is graph-replay/state corruption, not a tie"
    );
}

#[cfg(feature = "wgpu")]
pub fn spec_dims_tiny() -> SpecDims {
    SpecDims {
        h: 32,
        nh: 4,
        nkv: 2,
        hd: 8,
        inter: 64,
        vocab: 96,
        max_seq: 192,
        eps: 1e-5,
        rope_theta: 10000.0,
    }
}
