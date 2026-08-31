#![cfg(feature = "wgpu")]

use candle_core::Device;
use nv_models::qwen3_5_dense_wgpu::{Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_weights::WeightLoader;
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn qwen38_snapshot_dir() -> PathBuf {
    let d = std::env::var("NV_QWEN38_DIR").expect(
        "set NV_QWEN38_DIR to the Qwen3.8-27B-NVFP4 snapshot; this gate must never silently skip",
    );
    let p = PathBuf::from(&d);
    assert!(
        p.join("config.json").is_file(),
        "NV_QWEN38_DIR={d} has no config.json"
    );
    p
}

#[test]
#[ignore = "loads the 27B on wgpu; set NV_QWEN38_WGPU_TEST=1 -- free-runs 64 greedy steps feeding the HOST argmax of decode_step_logits and counts positions where the DEVICE argmax token disagrees; a nonzero count on any adapter is a device-argmax defect (the tf-ppl walk never calls the device argmax, so this seam is otherwise uncovered)"]
fn q3d_device_argmax_token_matches_host_argmax_of_the_same_logits_over_64_free_run_steps() {
    if std::env::var("NV_QWEN38_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_QWEN38_WGPU_TEST != 1");
        return;
    }
    let dir = qwen38_snapshot_dir();
    let cfg = Qwen3_5DenseConfig::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).expect("open weights");
    let max_seq: usize = std::env::var("NV_ARGMAX_AB_MAX_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    eprintln!("max_seq={max_seq}");
    let mut m = Qwen3_5DenseWgpu::from_loader(cfg, &loader, max_seq).expect("from_loader");
    drop(loader);

    let q = std::env::var("NV_QWEN38_PROMPT")
        .unwrap_or_else(|_| "What is the capital of France? Answer in one short sentence.".into());
    let text =
        format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let mut prompt = tok
        .encode(text.as_str(), false)
        .expect("encode chat prompt")
        .get_ids()
        .to_vec();
    if let Ok(pad) = std::env::var("NV_ARGMAX_AB_PAD_TO") {
        let n: usize = pad.parse().expect("NV_ARGMAX_AB_PAD_TO must be a usize");
        assert!(
            n >= prompt.len(),
            "NV_ARGMAX_AB_PAD_TO={n} < prompt {} -- padding never truncates the ask",
            prompt.len()
        );
        let filler = prompt[prompt.len() - 2];
        let ask: Vec<u32> = prompt.split_off(prompt.len().saturating_sub(24));
        while prompt.len() + ask.len() < n {
            prompt.push(filler);
        }
        prompt.extend_from_slice(&ask);
    }
    eprintln!(
        "prompt_tokens={} rest_mod_16={} (rest=len-1; 0 means zero prefill_step tail calls)",
        prompt.len(),
        (prompt.len() - 1) % 16
    );
    let first = m.prefill(&prompt).expect("prefill");

    let mut mismatches: Vec<(usize, u32, u32)> = Vec::new();
    let mut host_tok = first;
    let mut host_ids = vec![first];
    for i in 0..64usize {
        let (dev_tok, logits) = m.decode_step_logits(host_tok).expect("decode_step_logits");
        let host_arg = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite logits"))
            .map(|(j, _)| j as u32)
            .expect("nonempty logits");
        if dev_tok != host_arg {
            mismatches.push((i, dev_tok, host_arg));
        }
        host_tok = host_arg;
        host_ids.push(host_arg);
    }
    let out = tok.decode(&host_ids, false).unwrap_or_default();
    eprintln!(
        "Q3D-ARGMAX-AB steps=64 mismatches={} host_text={:?}",
        mismatches.len(),
        out
    );
    for (i, d, h) in mismatches.iter().take(8) {
        eprintln!("  step {i}: device={d} host={h}");
    }
    assert!(
        mismatches.is_empty(),
        "device argmax disagreed with host argmax of the same logits at {} of 64 steps; \
         first at step {} (device {} vs host {})",
        mismatches.len(),
        mismatches[0].0,
        mismatches[0].1,
        mismatches[0].2
    );
}
