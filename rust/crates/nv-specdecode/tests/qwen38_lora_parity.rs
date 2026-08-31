#![cfg(feature = "cuda")]

mod hub_dirs;

use candle_core::Device;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_specdecode::qwen38_mtp::{
    mtp_chain_depth_from_env, mtp_draft_dir_override_from_env, Qwen38DenseMtpHead,
    Qwen38MtpDecodeSession, Qwen38MtpSelfSpecEngine, MTP_WEIGHTS_FILE_NAME,
};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
mod common;
use common::render_real_chat_template_no_thinking;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";

const LORA_PARITY_PROMPTS_CREATIVE_SO_STEADY_STATE_IS_REAL_GENERATION: [&str; 2] = [
    "Write a short story about a lighthouse keeper who discovers the light attracts something \
     other than ships. Aim for three paragraphs.",
    "Describe a busy port market at dawn through the eyes of a cat that lives between the fish \
     stalls. Rich sensory detail, two paragraphs.",
];

const WARM_TOKENS_16_UNTIMED_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW: usize = 16;

const REF_MS_TOK_PARITY_BAND_5_PCT_BECAUSE_MERGED_WEIGHTS_RUN_BYTE_IDENTICAL_KERNEL_ROUTES: f64 =
    0.05;

fn merged_dir_from_env() -> PathBuf {
    let v = std::env::var("NV_Q38_LORA_MERGED_DIR").expect(
        "NV_Q38_LORA_MERGED_DIR must point at a lora-merged Qwen3.8 snapshot (build one with \
         nvk-lora-merge)",
    );
    let p = PathBuf::from(v);
    assert!(
        p.join("model.safetensors").is_file(),
        "NV_Q38_LORA_MERGED_DIR {} lacks model.safetensors",
        p.display()
    );
    p
}

fn load_model(dir: &Path, device: &Device) -> Qwen3Moe {
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = WeightLoader::open_dir(dir, device).expect("weights");
    Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, device)
        .expect("track-2 dense NVFP4 loader must load the snapshot")
}

fn junk_draft_stream(
    base: &Qwen3Moe,
    mtp: &Qwen38DenseMtpHead,
    k: usize,
    prompt: &[u32],
    max_new: usize,
    max_seq: usize,
    stop: &[u32],
) -> Vec<u32> {
    let mut session =
        Qwen38MtpDecodeSession::start(base, mtp, k, prompt, max_seq).expect("junk session");
    let junk = vec![0u32; k];
    let anchor = session.anchor();
    let mut generated = vec![anchor];
    if stop.contains(&anchor) {
        return generated;
    }
    'rounds: while generated.len() < max_new && session.round_fits() {
        let emitted = session
            .round_with_drafts_from_a_clairvoyant_test_oracle(&junk)
            .expect("junk round");
        for &t in emitted.iter() {
            generated.push(t);
            if stop.contains(&t) || generated.len() >= max_new {
                break 'rounds;
            }
        }
    }
    generated
}

#[test]
#[ignore = "loads TWO ~22 GB Qwen3.8-27B snapshots (base + lora-merged) on one card; set \
            NV_Q38_LORA=1 and NV_Q38_LORA_MERGED_DIR; interleaved same-session A/B: for each \
            prompt the eager m=1 reference decode and the MTP spec decode run on both models \
            back to back, each arm warmed untimed; the speed gate is on the reference arm \
            (merged weights are byte-layout-identical storage on identical kernel routes, so \
            ms/tok must match the base within noise); the spec arm gate is merged-model \
            self-consistency (junk drafts must emit the identical stream), acceptance-rate \
            changes vs base are EXPECTED because the weights changed"]
fn qwen38_lora_merged_decodes_at_base_speed_and_spec_is_self_consistent() {
    if std::env::var("NV_Q38_LORA").as_deref() != Ok("1") {
        panic!("set NV_Q38_LORA=1 to run (it must never silently skip)");
    }
    let base_dir = hub_dirs::snapshot(
        NVFP4_REPO,
        &[
            "config.json",
            "tokenizer.json",
            "chat_template.jinja",
            MTP_WEIGHTS_FILE_NAME,
        ],
    )
    .expect("base Qwen3.8-27B-NVFP4 snapshot with the MTP shard not found in the hub cache");
    let merged_dir = merged_dir_from_env();

    let tok = Tokenizer::from_file(base_dir.join("tokenizer.json")).expect("tokenizer.json");
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.token_to_id(s))
        .collect();
    assert!(!stop.is_empty(), "tokenizer lost the stop tokens");

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let base = load_model(&base_dir, &device);
    let merged = load_model(&merged_dir, &device);

    let base_mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &base_dir,
        &base,
        &device,
    )
    .expect("base MTP head");
    let merged_mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &merged_dir,
        &merged,
        &device,
    )
    .expect("merged MTP head");

    let k = mtp_chain_depth_from_env();
    let base_eng = Qwen38MtpSelfSpecEngine::new(&base, &base_mtp, k)
        .expect("base engine")
        .with_stop_ids(stop.clone());
    let merged_eng = Qwen38MtpSelfSpecEngine::new(&merged, &merged_mtp, k)
        .expect("merged engine")
        .with_stop_ids(stop.clone());

    let max_new: usize = std::env::var("NV_Q38_LORA_MAX_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let max_seq: usize = 2048;
    let warm = WARM_TOKENS_16_UNTIMED_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW;

    for (pi, question) in LORA_PARITY_PROMPTS_CREATIVE_SO_STEADY_STATE_IS_REAL_GENERATION
        .iter()
        .enumerate()
    {
        let prompt_text = render_real_chat_template_no_thinking(&base_dir, question);
        let prompt: Vec<u32> = tok
            .encode(prompt_text.as_str(), false)
            .expect("encode")
            .get_ids()
            .to_vec();

        base_eng
            .generate_reference(&prompt, warm, max_seq)
            .expect("base ref warm");
        let (base_ref_ids, base_ref_stats) = base_eng
            .generate_reference(&prompt, max_new, max_seq)
            .expect("base ref");

        merged_eng
            .generate_reference(&prompt, warm, max_seq)
            .expect("merged ref warm");
        let (merged_ref_ids, merged_ref_stats) = merged_eng
            .generate_reference(&prompt, max_new, max_seq)
            .expect("merged ref");

        base_eng
            .generate_greedy(&prompt, warm, max_seq)
            .expect("base spec warm");
        let (base_spec_ids, base_spec_stats) = base_eng
            .generate_greedy(&prompt, max_new, max_seq)
            .expect("base spec");

        merged_eng
            .generate_greedy(&prompt, warm, max_seq)
            .expect("merged spec warm");
        let (merged_spec_ids, merged_spec_stats) = merged_eng
            .generate_greedy(&prompt, max_new, max_seq)
            .expect("merged spec");

        let junk_ids = junk_draft_stream(
            &merged, &merged_mtp, k, &prompt, max_new, max_seq, &stop,
        );
        assert_eq!(
            merged_spec_ids, junk_ids,
            "prompt {pi}: merged-model spec decode is not self-consistent -- the emitted stream \
             must be the batched verify's own greedy continuation regardless of drafts"
        );

        let base_ref_ms = base_ref_stats.round_ms.iter().sum::<f64>()
            / base_ref_ids.len().max(1) as f64;
        let merged_ref_ms = merged_ref_stats.round_ms.iter().sum::<f64>()
            / merged_ref_ids.len().max(1) as f64;
        let base_spec_ms = (base_spec_stats.draft_ms + base_spec_stats.verify_ms)
            / base_spec_ids.len().max(1) as f64;
        let merged_spec_ms = (merged_spec_stats.draft_ms + merged_spec_stats.verify_ms)
            / merged_spec_ids.len().max(1) as f64;

        let first_div = base_spec_ids
            .iter()
            .zip(merged_spec_ids.iter())
            .take_while(|(a, b)| a == b)
            .count();

        eprintln!(
            "Q38 LORA PARITY basis=(base={}, merged={}, adapter folded offline via \
             nvk-lora-merge, k={k}, max_new={max_new}, prompt={pi} prompt_toks={}, template= \
             official no-thinking, greedy, warmed {warm} untimed, eager ref + eager spec): \
             base_ref_ms_tok={base_ref_ms:.2} merged_ref_ms_tok={merged_ref_ms:.2} \
             ref_delta={:+.2}% base_spec_ms_tok={base_spec_ms:.2} \
             merged_spec_ms_tok={merged_spec_ms:.2} base_accept={:.3} merged_accept={:.3} \
             base_new={} merged_new={} basevsmerged_first_divergence={first_div}",
            base_dir.display(),
            merged_dir.display(),
            prompt.len(),
            100.0 * (merged_ref_ms - base_ref_ms) / base_ref_ms,
            base_spec_stats.accept_rate(),
            merged_spec_stats.accept_rate(),
            base_ref_ids.len(),
            merged_ref_ids.len(),
        );

        let band = REF_MS_TOK_PARITY_BAND_5_PCT_BECAUSE_MERGED_WEIGHTS_RUN_BYTE_IDENTICAL_KERNEL_ROUTES;
        assert!(
            (merged_ref_ms - base_ref_ms).abs() / base_ref_ms <= band,
            "prompt {pi}: merged reference decode {merged_ref_ms:.2} ms/tok drifted more than \
             {:.0}% from base {base_ref_ms:.2} ms/tok; merged weights change no kernel route, \
             so this is a serving regression, not an adapter cost",
            band * 100.0
        );
    }
}
