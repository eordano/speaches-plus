#![cfg(feature = "cuda")]

#[path = "laguna_prompts.rs"]
mod prompts;
mod hub_snapshot;

use candle_core::Device;
use nv_models::laguna::{Laguna, LagunaConfig};
use nv_models::laguna_serve::{
    load_dflash_draft, spec_serve_loop, SpecServeEvent, SpecServeJob,
};
use prompts::LagunaEval;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;

const SERVE_MAX_NEW_64_ENOUGH_ROUNDS_TO_EXERCISE_PROPOSER_AND_VERIFY_GRAPHS: usize = 64;

const SERVE_NUM_SPEC_4_THE_DEFAULT_DFLASH_ROUND_SHAPE: usize = 4;

const SPEC_VS_M1_COMMON_PREFIX_16_BATCHED_VERIFY_LOGITS_MAY_FLIP_NEAR_TIES_LATER: usize = 16;

fn dflash_draft_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_DFLASH_DRAFT_DIR") {
        return PathBuf::from(d);
    }
    hub_snapshot::snapshot_of(
        "poolside/Laguna-XS-2.1-DFlash-NVFP4",
        &["config.json", "*.safetensors"],
    )
    .expect(
        "no hydrated poolside/Laguna-XS-2.1-DFlash-NVFP4 snapshot under the HF hub roots; set \
         NV_DFLASH_DRAFT_DIR",
    )
}

fn load_target(dir: &PathBuf, device: &Device) -> Arc<Laguna> {
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("read config");
    let config = LagunaConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qconfig =
        nv_weights::QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse quant config");
    let weights = nv_weights::WeightLoader::open_dir(dir, device).expect("open weights");
    Arc::new(
        Laguna::from_loader_quantized(config, &weights, &qconfig, device).expect("load target"),
    )
}

fn serve_once(
    target: &Arc<Laguna>,
    draft_dir: Option<&PathBuf>,
    device: &Device,
    prompt_ids: &[u32],
    max_seq: usize,
) -> (bool, Vec<u32>) {
    let draft = draft_dir.map(|d| load_dflash_draft(d, device).expect("load dflash draft"));
    let (job_tx, job_rx) = mpsc::channel::<SpecServeJob>();
    let (ready_tx, ready_rx) = mpsc::channel::<bool>();
    let loop_target = Arc::clone(target);
    let handle = std::thread::spawn(move || {
        spec_serve_loop(
            loop_target,
            draft,
            SERVE_NUM_SPEC_4_THE_DEFAULT_DFLASH_ROUND_SHAPE,
            max_seq,
            job_rx,
            ready_tx,
        )
    });
    let has_draft = ready_rx.recv().expect("serve loop must signal readiness");
    let (tok_tx, tok_rx) = mpsc::channel::<SpecServeEvent>();
    let job = SpecServeJob {
        prompt_ids: prompt_ids.to_vec(),
        prompt_text: String::new(),
        max_new: SERVE_MAX_NEW_64_ENOUGH_ROUNDS_TO_EXERCISE_PROPOSER_AND_VERIFY_GRAPHS,
        eos_ids: Vec::new(),
        emit: Box::new(move |ev| tok_tx.send(ev).is_ok()),
    };
    job_tx.send(job).expect("submit job");
    let mut tokens = Vec::new();
    let mut done = false;
    while let Ok(ev) = tok_rx.recv() {
        match ev {
            SpecServeEvent::Tokens(t) => tokens.extend(t),
            SpecServeEvent::Done => {
                done = true;
                break;
            }
            SpecServeEvent::Error(e) => panic!("serve loop returned an error: {e}"),
        }
    }
    assert!(done, "serve loop dropped the emit channel without Done");
    drop(job_tx);
    handle
        .join()
        .expect("serve loop thread panicked")
        .expect("serve loop returned Err");
    (has_draft, tokens)
}

#[test]
#[ignore = "loads the Laguna-XS-2.1 NVFP4 target plus the DFlash draft and drives \
            laguna_serve::spec_serve_loop end-to-end; set NV_LAGUNA_TEST=1 and \
            NV_LAGUNA_DFLASH=1 -- the spec-served stream must share a long greedy prefix with \
            the M=1 stream and run to budget, and the warmup+decode rounds are the live path \
            through LagunaStepGraph, DflashGraphProposer and LagunaVerifyGraph construction \
            that the failed-ctor guard protects"]
fn spec_serve_loop_with_the_dflash_draft_emits_the_same_greedy_stream_as_m1() {
    if std::env::var("NV_LAGUNA_TEST").is_err() || std::env::var("NV_LAGUNA_DFLASH").is_err() {
        eprintln!("set NV_LAGUNA_TEST=1 and NV_LAGUNA_DFLASH=1 to run");
        return;
    }
    let ev = LagunaEval::open().expect("laguna snapshot + prompt pack");
    eprintln!("{}", ev.describe());
    let device = Device::new_cuda(0).expect("cuda device");
    let target = load_target(&ev.dir, &device);
    let prompt_ids = ev.ids("openended-code").expect("pack prompt");
    let max_seq = prompt_ids.len()
        + 2 * SERVE_MAX_NEW_64_ENOUGH_ROUNDS_TO_EXERCISE_PROPOSER_AND_VERIFY_GRAPHS
        + 64;

    let (m1_has_draft, m1_tokens) = serve_once(&target, None, &device, &prompt_ids, max_seq);
    assert!(!m1_has_draft, "no-draft serve must report has_draft=false");
    assert!(
        m1_tokens.len() >= SERVE_MAX_NEW_64_ENOUGH_ROUNDS_TO_EXERCISE_PROPOSER_AND_VERIFY_GRAPHS / 2,
        "M=1 serve emitted only {} tokens",
        m1_tokens.len()
    );

    let draft_dir = dflash_draft_dir();
    let (spec_has_draft, spec_tokens) =
        serve_once(&target, Some(&draft_dir), &device, &prompt_ids, max_seq);
    assert!(
        spec_has_draft,
        "the DFlash draft was rejected at serve boot; the spec path under test never engaged"
    );
    assert!(
        spec_tokens.len()
            >= SERVE_MAX_NEW_64_ENOUGH_ROUNDS_TO_EXERCISE_PROPOSER_AND_VERIFY_GRAPHS / 2,
        "spec serve emitted only {} tokens",
        spec_tokens.len()
    );
    let common = spec_tokens
        .iter()
        .zip(m1_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();
    assert!(
        common >= SPEC_VS_M1_COMMON_PREFIX_16_BATCHED_VERIFY_LOGITS_MAY_FLIP_NEAR_TIES_LATER,
        "spec stream shares only a {common}-token prefix with the M=1 greedy stream \
         (spec {spec_tokens:?} vs m1 {m1_tokens:?}); an immediate divergence means the \
         accept/verify path is reading corrupt state, not flipping a near-tie -- batched-verify \
         logits and single-step logits are different gemm shapes with different reduction \
         orders, so full-stream identity is not the contract, an early split is"
    );
    eprintln!(
        "[spec-serve] emitted {} tokens, {common}-token common prefix with the M=1 stream",
        spec_tokens.len()
    );
}
