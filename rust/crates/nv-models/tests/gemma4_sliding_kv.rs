#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::{Gemma4, Gemma4Config};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn gemma4_nvfp4_snapshot_dir() -> PathBuf {
    PathBuf::from(std::env::var("NV_G4_SNAPSHOT").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots/e5ef03afa233c35cb000323ff098d4291e1dd07c",
            std::env::var("HOME").unwrap_or_default()
        )
    }))
}

fn argmax_last(logits: &Tensor, vocab: usize) -> u32 {
    let v: Vec<f32> = logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let last = &v[v.len() - vocab..];
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in last.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i as u32;
        }
    }
    best
}

fn generate(
    model: &Gemma4,
    device: &Device,
    ids: &[u32],
    n_new: usize,
    max_seq: usize,
) -> Vec<u32> {
    let vocab = model.config().vocab_size;
    let mut cache = model.new_kv_cache(max_seq).expect("kv cache");

    let seq = ids.len();
    let tokens = Tensor::from_vec(ids.to_vec(), (1usize, seq), device).unwrap();
    let pos: Vec<i32> = (0..seq as i32).collect();
    let pos = Tensor::from_vec(pos, seq, device).unwrap();
    let logits = model
        .forward_with_cache(&tokens, &pos, &mut cache)
        .expect("prefill");
    let mut next = argmax_last(&logits, vocab);

    let mut out = vec![next];
    for i in 0..n_new {
        let p = seq + i;
        let tok = Tensor::from_vec(vec![next], (1usize, 1usize), device).unwrap();
        let posn = Tensor::from_vec(vec![p as i32], 1usize, device).unwrap();
        let logits = model
            .forward_with_cache(&tok, &posn, &mut cache)
            .expect("decode");
        next = argmax_last(&logits, vocab);
        out.push(next);
    }
    out
}

#[test]
fn sliding_kv_matches_full_storage_past_window() {
    let dir = gemma4_nvfp4_snapshot_dir();
    if !dir.is_dir() {
        eprintln!("skip: snapshot dir missing {}", dir.display());
        return;
    }
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let cfg = Gemma4Config::from_hf_json_str(&raw).unwrap();
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).unwrap();
    let weights = WeightLoader::open_dir(&dir, &device).unwrap();
    let model = Gemma4::from_loader_quantized(cfg.clone(), &weights, &qcfg, &device).unwrap();
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();

    let mut ids: Vec<u32> = tok
        .encode(
            "The history of computing, from the abacus onward, is",
            false,
        )
        .unwrap()
        .get_ids()
        .to_vec();
    ids.insert(0, 2);

    let n_new = 1400usize;
    let max_seq = ids.len() + n_new + 8;

    std::env::remove_var("NV_KV_NO_SLIDING");
    let probe_n = 120usize;
    let probe_a = generate(&model, &device, &ids, probe_n, max_seq);
    let probe_b = generate(&model, &device, &ids, probe_n, max_seq);
    let probe_div = probe_a.iter().zip(probe_b.iter()).position(|(a, b)| a != b);
    eprintln!(
        "determinism probe (same sliding config x2, {probe_n} tok): first divergence = {:?}",
        probe_div
    );

    eprintln!("generating {n_new} tokens with the sliding KV cache...");
    std::env::remove_var("NV_KV_NO_SLIDING");
    let sliding = generate(&model, &device, &ids, n_new, max_seq);

    eprintln!("generating {n_new} tokens with full storage (NV_KV_NO_SLIDING)...");
    std::env::set_var("NV_KV_NO_SLIDING", "1");
    let full = generate(&model, &device, &ids, n_new, max_seq);
    std::env::remove_var("NV_KV_NO_SLIDING");

    let text = tok
        .decode(&sliding[..200.min(sliding.len())], false)
        .unwrap_or_default();
    eprintln!("sliding[..200] decoded: {text:?}");

    let uniq: std::collections::HashSet<u32> = sliding.iter().copied().collect();
    assert!(
        uniq.len() >= 5,
        "sliding output degenerate ({} unique of {} tokens)",
        uniq.len(),
        sliding.len()
    );

    assert_eq!(sliding.len(), full.len(), "length mismatch sliding vs full");
    let first_div = sliding.iter().zip(full.iter()).position(|(a, b)| a != b);
    eprintln!(
        "sliding vs full storage: first divergence = {:?} (compaction happens past {})",
        first_div,
        1024 + 256
    );

    if probe_div.is_none() {
        assert!(
            first_div.is_none(),
            "deterministic forward, yet sliding KV diverged from full at token {:?} \
             (sliding={:?} full={:?}) -- bug in the sliding cache",
            first_div,
            first_div.map(|i| sliding[i]),
            first_div.map(|i| full[i]),
        );
        eprintln!("OK: forward is deterministic AND {n_new} tokens identical sliding vs full");
    } else {
        eprintln!(
            "NOTE: eager forward is non-deterministic (probe diverged at {:?}); \
             relying on coherence + the no-earlier-than-probe check",
            probe_div
        );
        if let (Some(fd), Some(pd)) = (first_div, probe_div) {
            assert!(
                fd + 50 >= pd,
                "sliding vs full diverged at {fd}, far earlier than two identical runs ({pd}) \
                 -- suggests a real sliding-cache bug, not just nondeterminism"
            );
        }
        eprintln!("OK: sliding KV coherent past the window; divergence consistent with forward nondeterminism");
    }
}
