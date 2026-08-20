#![cfg(feature = "wgpu")]

use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_wgpu::{EmbedRowSplice, Gemma4Wgpu, HostWeights};
mod common;
use common::gemma4_wgpu_host_weights as host_weights;
use common::LcgShift32Centered0p1I8 as Lcg;

const TINY_CONFIG: &str = r#"{
  "text_config": {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 32,
    "global_head_dim": 32,
    "vocab_size": 512,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "sliding_window": 8,
    "final_logit_softcapping": 0.0,
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
    "attention_k_eq_v": false,
    "hidden_activation": "gelu_pytorch_tanh",
    "num_kv_shared_layers": 0,
    "rope_parameters": {
      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
      "sliding_attention": {"rope_theta": 10000.0}
    }
  },
  "tie_word_embeddings": true
}"#;

fn ctx_or_skip() -> bool {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("[g4w-verify] adapter: {}", ctx.summary());
            true
        }
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            false
        }
    }
}

fn tiny_config() -> Gemma4Config {
    Gemma4Config::from_hf_json_str(TINY_CONFIG).expect("tiny config parses")
}

fn tiny_model(max_seq: usize) -> Gemma4Wgpu {
    let config = tiny_config();
    let weights = host_weights(&config, 0x5eed_1234);
    Gemma4Wgpu::new(config, &weights, max_seq).expect("tiny gemma4 dense wgpu model")
}

fn stepped_replay(m: &mut Gemma4Wgpu, batch: &[u32]) -> Vec<u32> {
    let pos0 = m.current_pos();
    let out: Vec<u32> = batch
        .iter()
        .map(|&t| m.decode_step(t).expect("decode_step"))
        .collect();
    m.truncate_to(pos0).expect("truncate back to the round start");
    out
}

fn scaled_embed_row(weights: &HostWeights, hidden: usize, token: u32) -> Vec<u16> {
    let scale = (hidden as f32).sqrt();
    let base = token as usize * hidden;
    weights.embed[base..base + hidden]
        .iter()
        .map(|&b| {
            let v = half::bf16::from_bits(b).to_f32() * scale;
            half::bf16::from_f32(v).to_bits()
        })
        .collect()
}

#[test]
fn verify_chain_argmax_is_bit_identical_to_stepped_decode_at_random_depths() {
    if !ctx_or_skip() {
        return;
    }
    let mut m = tiny_model(96);
    let cap = m.verify_max_rows();
    assert!(
        cap >= 2,
        "verify epilogue disabled (rows {cap}); the comparison would be vacuous"
    );
    assert_eq!(
        cap,
        m.prefill_chunk_len().min(9),
        "verify rows must follow the m-row prefill width capped at the longest spec chain"
    );
    let vocab = m.config().vocab_size;
    let window = m.config().sliding_window;
    let prefix: Vec<u32> = (0..21u32).map(|i| (i * 37 + 5) % vocab as u32).collect();
    assert!(
        prefix.len() > window,
        "prefix must cross the sliding window ({} vs {window})",
        prefix.len()
    );
    for &t in &prefix {
        m.decode_step(t).expect("prime the prefix");
    }

    let mut rng = Lcg(0xc0ffee);
    let mut rounds_checked = 0usize;
    for round in 0..10 {
        let k = 1 + round % cap;
        let batch: Vec<u32> = (0..k).map(|_| rng.token(vocab)).collect();
        let pos0 = m.current_pos();
        let va = m.verify_chain(&batch).expect("verify_chain");
        assert_eq!(m.current_pos(), pos0, "verify_chain must not move pos");
        assert_eq!(va.len(), k);
        let sa = stepped_replay(&mut m, &batch);
        assert_eq!(
            va, sa,
            "round {round}: m-row verify argmax != per-token stepped argmax"
        );
        let commit = 1 + (round * 7) % k;
        m.advance(commit).expect("advance the accepted prefix");
        assert_eq!(m.current_pos(), pos0 + commit);
        rounds_checked += 1;
    }
    assert_eq!(rounds_checked, 10);
}

#[test]
fn verify_chain_row_logits_are_bit_identical_to_stepped_decode_logits() {
    if !ctx_or_skip() {
        return;
    }
    let mut m = tiny_model(96);
    let cap = m.verify_max_rows();
    assert!(cap >= 2, "verify epilogue disabled (rows {cap})");
    let vocab = m.config().vocab_size;
    let prefix: Vec<u32> = (0..13u32).map(|i| (i * 53 + 11) % vocab as u32).collect();
    for &t in &prefix {
        m.decode_step(t).expect("prime the prefix");
    }
    let k = cap.min(4);
    let mut rng = Lcg(0xabcd_ef01);
    let batch: Vec<u32> = (0..k).map(|_| rng.token(vocab)).collect();
    let pos0 = m.current_pos();
    m.verify_chain(&batch).expect("verify_chain");
    let verify_rows: Vec<Vec<u32>> = (0..k)
        .map(|r| {
            m.verify_row_logits(r)
                .expect("verify row logits")
                .into_iter()
                .map(f32::to_bits)
                .collect()
        })
        .collect();
    let mut stepped_rows: Vec<Vec<u32>> = Vec::new();
    for &t in &batch {
        let (_tok, logits) = m.decode_step_logits(t).expect("decode_step_logits");
        stepped_rows.push(logits.into_iter().map(f32::to_bits).collect());
    }
    m.truncate_to(pos0).expect("truncate back");
    let spread = {
        let f: Vec<f32> = stepped_rows[0].iter().map(|b| f32::from_bits(*b)).collect();
        f.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b))
            - f.iter().fold(f32::INFINITY, |a, b| a.min(*b))
    };
    assert!(
        spread > 1e-3,
        "logits are flat ({spread}); the comparison would not discriminate"
    );
    for r in 0..k {
        assert_eq!(verify_rows[r].len(), vocab);
        let diff = verify_rows[r]
            .iter()
            .zip(stepped_rows[r].iter())
            .filter(|(a, b)| a != b)
            .count();
        let worst = verify_rows[r]
            .iter()
            .zip(stepped_rows[r].iter())
            .map(|(a, b)| (f32::from_bits(*a) - f32::from_bits(*b)).abs())
            .fold(0f32, f32::max);
        assert_eq!(
            diff, 0,
            "verify row {r}: {diff} of {vocab} logit lanes differ from the stepped decode \
             (worst |delta| {worst:e})"
        );
    }
    eprintln!("[g4w-verify] {k} rows x {vocab} logit lanes bit-identical to stepped decode");
}

#[test]
fn verify_chain_rejects_batches_outside_its_row_capacity() {
    if !ctx_or_skip() {
        return;
    }
    let mut m = tiny_model(64);
    let cap = m.verify_max_rows();
    assert!(cap >= 2);
    m.decode_step(3).expect("prime");
    let pos = m.current_pos();
    assert!(m.verify_chain(&[]).is_err(), "empty batch must fail");
    let long: Vec<u32> = vec![1; cap + 1];
    assert!(m.verify_chain(&long).is_err(), "oversized batch must fail");
    let vocab = m.config().vocab_size as u32;
    assert!(
        m.verify_chain(&[vocab]).is_err(),
        "out-of-vocab token must fail"
    );
    assert_eq!(m.current_pos(), pos, "a rejected batch must not move pos");
}

#[test]
fn embed_row_prefill_without_splices_is_bit_identical_to_plain_prefill() {
    if !ctx_or_skip() {
        return;
    }
    let mut m = tiny_model(96);
    let mm = m.prefill_chunk_len();
    assert!(mm >= 2, "chunked prefill disabled (m={mm})");
    let vocab = m.config().vocab_size;
    let prompt: Vec<u32> = (0..(2 * mm + 3) as u32)
        .map(|i| (i * 29 + 7) % vocab as u32)
        .collect();
    let (last, rest) = prompt.split_last().expect("non-empty prompt");

    m.reset();
    let done_a = m.prefill_tokens(rest).expect("plain prefill_tokens");
    for t in &rest[done_a..] {
        m.prefill_step(*t).expect("prefill step");
    }
    let (tok_a, logits_a) = m.decode_step_logits(*last).expect("decode");

    m.reset();
    let done_b = m
        .prefill_tokens_with_embed_rows(rest, &[])
        .expect("embed-row prefill with no splices");
    assert_eq!(done_b, rest.len(), "the embed-row path must consume all rows");
    let (tok_b, logits_b) = m.decode_step_logits(*last).expect("decode");

    assert_eq!(tok_a, tok_b, "next token moved with an empty splice list");
    let worst = logits_a
        .iter()
        .zip(logits_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        logits_a.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            == logits_b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "empty-splice prefill is not bit-identical to plain prefill (worst {worst:e})"
    );
}

#[test]
fn splicing_the_models_own_embedding_rows_is_bit_identical_to_plain_prefill() {
    if !ctx_or_skip() {
        return;
    }
    let config = tiny_config();
    let weights = host_weights(&config, 0x5eed_1234);
    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let mut m = Gemma4Wgpu::new(config, &weights, 96).expect("tiny model");
    let mm = m.prefill_chunk_len();
    assert!(mm >= 2, "chunked prefill disabled (m={mm})");
    let prompt: Vec<u32> = (0..(2 * mm + 3) as u32)
        .map(|i| (i * 29 + 7) % vocab as u32)
        .collect();
    let (last, rest) = prompt.split_last().expect("non-empty prompt");

    m.reset();
    m.prefill_tokens_with_embed_rows(rest, &[])
        .expect("reference prefill");
    let (tok_a, logits_a) = m.decode_step_logits(*last).expect("decode");

    let splice_start = mm - 1;
    let splice_len = 3usize.min(rest.len() - splice_start);
    assert!(splice_len >= 2, "the splice must cross a chunk boundary");
    let mut rows_bf16: Vec<u16> = Vec::with_capacity(splice_len * hidden);
    for i in 0..splice_len {
        rows_bf16.extend(scaled_embed_row(&weights, hidden, rest[splice_start + i]));
    }
    let splices = vec![EmbedRowSplice {
        position: splice_start,
        rows_bf16,
    }];

    m.reset();
    m.prefill_tokens_with_embed_rows(rest, &splices)
        .expect("spliced prefill");
    let (tok_b, logits_b) = m.decode_step_logits(*last).expect("decode");

    assert_eq!(
        tok_a, tok_b,
        "splicing the model's own scaled embedding rows changed the next token"
    );
    let worst = logits_a
        .iter()
        .zip(logits_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert_eq!(
        logits_a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        logits_b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "spliced own-embedding rows are not bit-identical to the gathered rows (worst {worst:e})"
    );
    eprintln!(
        "[g4w-splice] {splice_len} spliced rows from position {splice_start} across m={mm} chunks \
         reproduce the gathered rows bit-for-bit"
    );
}

fn real_dense_snapshot() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("NV_CHAT_MODEL_DIR") {
        let p = std::path::PathBuf::from(d);
        if p.join("config.json").exists() {
            return p;
        }
    }
    let hub = std::env::var("HF_HUB_CACHE").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface/hub",
            std::env::var("HOME").expect("HOME")
        )
    });
    let base = std::path::PathBuf::from(hub)
        .join("models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("no snapshot dir at {}: {e}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists())
        .expect("no snapshot with config.json")
}

#[test]
#[ignore = "loads the real Gemma-4-31B NVFP4 checkpoint; set NV_G4D_VERIFY_TEST=1"]
fn real_gemma4_31b_verify_chain_matches_stepped_decode() {
    if std::env::var("NV_G4D_VERIFY_TEST").as_deref() != Ok("1") {
        panic!("set NV_G4D_VERIFY_TEST=1 to run this GPU test (it must never silently skip)");
    }
    assert!(ctx_or_skip(), "real-weights test needs a wgpu adapter");
    let dir = real_dense_snapshot();
    eprintln!("[g4w-verify-real] loading {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).unwrap();
    let vocab = config.vocab_size;
    let mut m = Gemma4Wgpu::new(config, &host, 1024).expect("build the real dense wgpu model");
    drop(host);
    let cap = m.verify_max_rows();
    assert!(cap >= 2, "verify epilogue disabled on the real checkpoint (rows {cap})");

    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let prompt = "The measurement of record for this repository is the pipeline comparison \
                  document, and every number in it carries its basis.";
    let ids: Vec<u32> = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();
    assert!(ids.len() > 8, "prompt is too short to prime the cache");
    let mut next = m.prefill(&ids).expect("prefill the prompt");

    let mut rng = Lcg(0xfeed_beef);
    let mut checked = 0usize;
    for round in 0..4 {
        let k = 1 + round % cap;
        let mut batch = vec![next];
        while batch.len() < k {
            batch.push(rng.token(vocab));
        }
        let pos0 = m.current_pos();
        let va = m.verify_chain(&batch).expect("verify_chain");
        assert_eq!(m.current_pos(), pos0, "verify_chain must not move pos");
        let sa = stepped_replay(&mut m, &batch);
        assert_eq!(
            va, sa,
            "round {round}: real-weights m-row verify argmax != per-token stepped argmax"
        );
        next = va[0];
        m.advance(1).expect("advance one accepted row");
        checked += 1;
    }
    assert_eq!(checked, 4);
    eprintln!("[g4w-verify-real] {checked} rounds argmax-identical up to {cap} rows");
}
