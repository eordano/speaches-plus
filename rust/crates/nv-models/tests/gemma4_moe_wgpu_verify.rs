#![cfg(feature = "wgpu")]

mod common;
use common::have_gpu;
use common::HIDDEN_64 as HIDDEN;
use common::real_snapshot;
use common::tiny_config_json;
use common::VOCAB_160 as VOCAB;
use common::WINDOW;
use candle_core::{Device, Tensor};
use nv_models::gemma4::LayerType;
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::{EmbedRowSplice, Gemma4MoeWgpu};
use nv_weights::WeightLoader;
use std::collections::HashMap;
use common::LcgTop24TwoSided as Lcg;
use common::TempDir;
use common::norm_tensor;
use common::rand_tensor_f32_shape as rand_tensor;
use common::tensors_for_cfg;

fn temp_dir(tag: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("g4m_vf_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

const SEED: u64 = 0x9e37_79b9_7f4a_7c15;

fn tiny_model(tag: &str, max_seq: usize) -> (Gemma4MoeWgpu, HashMap<String, Tensor>, TempDir) {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let dir = temp_dir(tag);
    let st = dir.0.join("model.safetensors");
    let tensors = tensors_for_cfg(&cfg, SEED);
    candle_core::safetensors::save(&tensors, &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
    let gpu = Gemma4MoeWgpu::from_loader(cfg, &loader, max_seq).expect("build wgpu model");
    (gpu, tensors, dir)
}

fn stepped_replay(m: &mut Gemma4MoeWgpu, batch: &[u32]) -> Vec<u32> {
    let pos0 = m.current_pos();
    let out: Vec<u32> = batch
        .iter()
        .map(|&t| m.decode_step(t).expect("decode_step"))
        .collect();
    m.truncate_to(pos0).expect("truncate back to the round start");
    out
}

fn scaled_embed_row(tensors: &HashMap<String, Tensor>, token: u32) -> Vec<u16> {
    let e = tensors["model.language_model.embed_tokens.weight"]
        .to_vec2::<f32>()
        .unwrap();
    let scale = (HIDDEN as f32).sqrt();
    e[token as usize]
        .iter()
        .map(|v| {
            let stored = half::bf16::from_f32(*v).to_f32();
            half::bf16::from_f32(stored * scale).to_bits()
        })
        .collect()
}

#[test]
fn verify_chain_argmax_is_bit_identical_to_stepped_decode_at_random_depths() {
    if !have_gpu() {
        return;
    }
    let (mut m, _t, _d) = tiny_model("depths", 60);
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
    let prefix: Vec<u32> = (0..15u32).map(|i| (i * 37 + 5) % VOCAB as u32).collect();
    assert!(
        prefix.len() > WINDOW,
        "prefix must cross the sliding window ({} vs {WINDOW})",
        prefix.len()
    );
    for &t in &prefix {
        m.decode_step(t).expect("prime the prefix");
    }

    let mut rng = Lcg(0xc0ffee);
    let mut rounds = 0usize;
    for round in 0..8 {
        let k = 1 + round % cap;
        let batch: Vec<u32> = (0..k).map(|_| rng.token(VOCAB)).collect();
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
        rounds += 1;
    }
    assert_eq!(rounds, 8);
}

#[test]
fn verify_chain_row_logits_are_bit_identical_to_stepped_decode_logits() {
    if !have_gpu() {
        return;
    }
    let (mut m, _t, _d) = tiny_model("logits", 60);
    let cap = m.verify_max_rows();
    assert!(cap >= 2, "verify epilogue disabled (rows {cap})");
    let prefix: Vec<u32> = (0..11u32).map(|i| (i * 53 + 11) % VOCAB as u32).collect();
    for &t in &prefix {
        m.decode_step(t).expect("prime the prefix");
    }
    let k = cap.min(4);
    let mut rng = Lcg(0xabcd_ef01);
    let batch: Vec<u32> = (0..k).map(|_| rng.token(VOCAB)).collect();
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
    let first: Vec<f32> = stepped_rows[0].iter().map(|b| f32::from_bits(*b)).collect();
    let spread = first.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b))
        - first.iter().fold(f32::INFINITY, |a, b| a.min(*b));
    assert!(
        spread > 1e-3,
        "logits are flat ({spread}); the comparison would not discriminate"
    );
    for r in 0..k {
        assert_eq!(verify_rows[r].len(), VOCAB);
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
            "verify row {r}: {diff} of {VOCAB} logit lanes differ from the stepped decode \
             (worst |delta| {worst:e})"
        );
    }
    eprintln!("[g4m-verify] {k} rows x {VOCAB} logit lanes bit-identical to stepped decode");
}

#[test]
fn verify_chain_rejects_batches_outside_its_row_capacity() {
    if !have_gpu() {
        return;
    }
    let (mut m, _t, _d) = tiny_model("reject", 40);
    let cap = m.verify_max_rows();
    assert!(cap >= 2);
    m.decode_step(3).expect("prime");
    let pos = m.current_pos();
    assert!(m.verify_chain(&[]).is_err(), "empty batch must fail");
    let long: Vec<u32> = vec![1; cap + 1];
    assert!(m.verify_chain(&long).is_err(), "oversized batch must fail");
    assert!(
        m.verify_chain(&[VOCAB as u32]).is_err(),
        "out-of-vocab token must fail"
    );
    assert_eq!(m.current_pos(), pos, "a rejected batch must not move pos");
}

#[test]
fn embed_row_prefill_without_splices_is_bit_identical_to_plain_prefill() {
    if !have_gpu() {
        return;
    }
    let (mut m, _t, _d) = tiny_model("nosplice", 60);
    let mm = m.prefill_chunk_len();
    assert!(mm >= 2, "chunked prefill disabled (m={mm})");
    let prompt: Vec<u32> = (0..(2 * mm + 3) as u32)
        .map(|i| (i * 29 + 7) % VOCAB as u32)
        .collect();
    let (last, rest) = prompt.split_last().expect("non-empty prompt");

    m.reset().unwrap();
    let done_a = m.prefill_tokens(rest).expect("plain prefill_tokens");
    for t in &rest[done_a..] {
        m.prefill_step(*t).expect("prefill step");
    }
    let (tok_a, logits_a) = m.decode_step_logits(*last).expect("decode");

    m.reset().unwrap();
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
    assert_eq!(
        logits_a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        logits_b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "empty-splice prefill is not bit-identical to plain prefill (worst {worst:e})"
    );
}

#[test]
fn splicing_the_models_own_embedding_rows_is_bit_identical_to_plain_prefill() {
    if !have_gpu() {
        return;
    }
    let (mut m, tensors, _d) = tiny_model("splice", 60);
    let mm = m.prefill_chunk_len();
    assert!(mm >= 2, "chunked prefill disabled (m={mm})");
    let prompt: Vec<u32> = (0..(2 * mm + 3) as u32)
        .map(|i| (i * 29 + 7) % VOCAB as u32)
        .collect();
    let (last, rest) = prompt.split_last().expect("non-empty prompt");

    m.reset().unwrap();
    m.prefill_tokens_with_embed_rows(rest, &[])
        .expect("reference prefill");
    let (tok_a, logits_a) = m.decode_step_logits(*last).expect("decode");

    let splice_start = mm - 1;
    let splice_len = 3usize.min(rest.len() - splice_start);
    assert!(splice_len >= 2, "the splice must cross a chunk boundary");
    let mut rows_bf16: Vec<u16> = Vec::with_capacity(splice_len * HIDDEN);
    for i in 0..splice_len {
        rows_bf16.extend(scaled_embed_row(&tensors, rest[splice_start + i]));
    }
    let splices = vec![EmbedRowSplice {
        position: splice_start,
        rows_bf16,
    }];

    m.reset().unwrap();
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
        "[g4m-splice] {splice_len} spliced rows from position {splice_start} across m={mm} chunks \
         reproduce the gathered rows bit-for-bit"
    );
}

#[test]
#[ignore = "loads the real gemma-4-26B-A4B checkpoint; set NV_GEMMA4_MOE_VERIFY_TEST=1"]
fn real_gemma4_moe_verify_chain_matches_stepped_decode() {
    if std::env::var("NV_GEMMA4_MOE_VERIFY_TEST").as_deref() != Ok("1") {
        panic!("set NV_GEMMA4_MOE_VERIFY_TEST=1 to run this GPU test (it must never silently skip)");
    }
    if !have_gpu() {
        panic!("real-weights test needs a wgpu adapter");
    }
    let snap = real_snapshot();
    let mut cfg = Gemma4MoeConfig::from_hf_json_file(&snap.join("config.json")).unwrap();
    if let Some(n) = std::env::var("NV_GEMMA4_MOE_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        assert!(n > 0 && n <= cfg.base.num_hidden_layers);
        cfg.base.num_hidden_layers = n;
        cfg.base.layer_types.truncate(n);
        eprintln!("[real] TRUNCATED to the first {n} layers (partial load)");
    }
    let vocab = cfg.base.vocab_size;
    let loader = WeightLoader::open_dir(&snap, &Device::Cpu).unwrap();
    let mut m = Gemma4MoeWgpu::from_loader(cfg, &loader, 512).expect("build the real moe model");
    let cap = m.verify_max_rows();
    assert!(
        cap >= 2,
        "verify epilogue disabled on the real checkpoint (rows {cap})"
    );

    let prefix: Vec<u32> = (0..33u32).map(|i| (i * 1031 + 17) % vocab as u32).collect();
    let mut next = m.prefill(&prefix).expect("prefill the prefix");
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
    eprintln!("[g4m-verify-real] {checked} rounds argmax-identical up to {cap} rows");
}
