#![cfg(feature = "cuda")]

mod hub_dirs;

use candle_core::Device;
use minijinja::{context, Environment, Value as JinjaValue};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_specdecode::qwen38_mtp::{
    mtp_draft_dir_override_from_env, Qwen38DenseMtpHead, Qwen38MtpGraphedDecodeSession,
    MTP_WEIGHTS_FILE_NAME,
};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::Path;
use tokenizers::Tokenizer;
mod common;
use common::raise_exception;
use common::render_real_chat_template_no_thinking;
use common::strftime_now;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";

const PROFILE_MAX_SEQ: usize = 2048;
const PROFILE_WARM_ROUNDS: usize = 6;
const PROFILE_TIMED_ROUNDS: usize = 12;
const PROFILE_ARMED_CHAINS: usize = 3;

const PROFILE_PROMPT_IS_THE_SWEEP_DOC_CHANGELOG_BRACKET: &str =
    "Continue this changelog with four more entries in exactly the same format, for versions \
     1.2.5 through 1.2.8:\n\n## 1.2.1\n- Fixed a crash when the config file is missing.\n- \
     Improved startup time by caching the model index.\n\n## 1.2.2\n- Fixed a crash when the \
     audio device is unplugged.\n- Improved startup time by lazy-loading the tokenizer.\n\n## \
     1.2.3\n- Fixed a crash when the network is unreachable.\n- Improved startup time by \
     deferring the license check.\n\n## 1.2.4\n- Fixed a crash when the cache directory is \
     read-only.\n- Improved startup time by precompiling the shaders.";

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 checkpoint; set NV_Q38_MTP=1; attributes the \
            per-step drafter cost (embed+fc / attn / mlp / lm_head / argmax readback) through \
            decode_prof, k from NV_MTP_K (default 7)"]
fn qwen38_mtp_draft_step_segment_attribution_on_the_graphed_session() {
    if std::env::var("NV_Q38_MTP").as_deref() != Ok("1") {
        panic!("set NV_Q38_MTP=1 to run (it must never silently skip)");
    }
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
    let k = std::env::var("NV_MTP_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(7);

    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("track-2 dense NVFP4 loader must load Qwen3.8-27B before this profile can run");
    drop(weights);
    let mut engine = nv_models::graph_engine::GraphedQwen3Moe::new(base, &device, PROFILE_MAX_SEQ)
        .expect("graphed engine over the dense arm");
    let mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &dir,
        engine.underlying(),
        &device,
    )
    .expect("MTP head loads from the shipped model_mtp.safetensors");

    let prompt_text =
        render_real_chat_template_no_thinking(&dir, PROFILE_PROMPT_IS_THE_SWEEP_DOC_CHANGELOG_BRACKET);
    let prompt: Vec<u32> = tok
        .encode(prompt_text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();

    let mut run_arm = |engine: &mut nv_models::graph_engine::GraphedQwen3Moe,
                       arm: &str,
                       armed_chains: usize|
     -> (Vec<u32>, f64) {
        {
            let mut warm = Qwen38MtpGraphedDecodeSession::start(engine, &mtp, k, &prompt)
                .expect("graphed mtp warm session start");
            for _ in 0..PROFILE_WARM_ROUNDS {
                warm.round().expect("warm round");
            }
        }
        let mut session = Qwen38MtpGraphedDecodeSession::start(engine, &mtp, k, &prompt)
            .expect("graphed mtp session start");
        let mut ids: Vec<u32> = vec![session.anchor()];
        for _ in 0..PROFILE_TIMED_ROUNDS {
            if !session.round_fits() {
                break;
            }
            ids.extend(session.round().expect("timed round"));
        }
        let rounds = session.stats.rounds;
        assert!(rounds > 0, "no timed rounds ran; the profile measured nothing");
        let draft_ms_per_round = session.stats.draft_ms / rounds as f64;
        eprintln!(
            "[q38-mtp-draftprof] arm={arm} k={k} prompt_toks={} rounds={rounds} \
             draft_ms_per_round={draft_ms_per_round:.3} draft_ms_per_step={:.3} \
             tokens_per_round={:.2} accept={:.3} \
             basis=(model=unsloth/Qwen3.8-27B-NVFP4 graphed-session doc-changelog-prompt)",
            prompt.len(),
            draft_ms_per_round / k as f64,
            session.stats.tokens_per_round(),
            session.stats.accept_rate(),
        );
        if armed_chains > 0 {
            unsafe {
                std::env::set_var("NV_PROF_DECODE", "1");
            }
            for _ in 0..armed_chains {
                session
                    .profile_one_draft_chain_then_rewind_arming_nv_prof_decode_whose_every_lap_syncs()
                    .expect("armed draft chain profile");
            }
            unsafe {
                std::env::remove_var("NV_PROF_DECODE");
            }
        }
        (ids, draft_ms_per_round)
    };

    let (baseline_ids, baseline_ms) = run_arm(&mut engine, "baseline", PROFILE_ARMED_CHAINS);

    unsafe {
        std::env::set_var(nv_specdecode::qwen38_mtp::NV_Q38_DRAFT_FAST_ENV, "1");
    }
    let (fast_ids, fast_ms) = run_arm(&mut engine, "draft_fast", 0);
    unsafe {
        std::env::remove_var(nv_specdecode::qwen38_mtp::NV_Q38_DRAFT_FAST_ENV);
    }

    assert_eq!(
        baseline_ids, fast_ids,
        "NV_Q38_DRAFT_FAST=1 must be bit-exact: the device-chained tokens are the same argmax \
         of the same logits, so any stream difference is a chaining bug"
    );
    eprintln!(
        "[q38-mtp-draftprof] arm=DELTA k={k} baseline_draft_ms_per_round={baseline_ms:.3} \
         fast_draft_ms_per_round={fast_ms:.3} speedup={:.3}x off_pct={:.1}",
        baseline_ms / fast_ms.max(1e-9),
        100.0 * (baseline_ms - fast_ms) / baseline_ms.max(1e-9),
    );
}
