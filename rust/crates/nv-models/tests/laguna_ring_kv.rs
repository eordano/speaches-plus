#![cfg(feature = "cuda")]

mod common;
use common::laguna_chunked_prefill as chunked_prefill;
#[path = "laguna_prompts.rs"]
mod prompts;
use prompts::{assert_publishable, LagunaEval};

use candle_core::{Device, Tensor};
use nv_models::laguna::{Laguna, LagunaConfig, LagunaKvCache};

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn lockstep_compare(
    model: &Laguna,
    device: &Device,
    prompt_ids: &[u32],
    steps: usize,
    label: &str,
) -> (usize, usize, Vec<u32>) {
    let cfg = model.config();
    let max_len = prompt_ids.len() + steps + 1;
    let mut ring_cache = LagunaKvCache::new_with_mode(cfg, max_len, device, model.dtype(), true)
        .expect("ring cache");
    let mut legacy_cache = LagunaKvCache::new_with_mode(cfg, max_len, device, model.dtype(), false)
        .expect("legacy cache");

    let ring_last = chunked_prefill(model, &mut ring_cache, prompt_ids, device);
    let legacy_last = chunked_prefill(model, &mut legacy_cache, prompt_ids, device);
    let mut agree = 0usize;
    let mut total = 0usize;
    let ring_a = argmax(&ring_last);
    let legacy_a = argmax(&legacy_last);
    total += 1;
    if ring_a == legacy_a {
        agree += 1;
    } else {
        eprintln!("[{label}] prefill argmax mismatch: ring {ring_a} legacy {legacy_a}");
    }

    let mut next = legacy_a;
    let mut reference: Vec<u32> = Vec::new();
    for step in 0..steps {
        reference.push(next);
        let pos = (prompt_ids.len() + step) as i32;
        let tt = Tensor::from_vec(vec![next], (1usize, 1usize), device).unwrap();
        let pp = Tensor::from_vec(vec![pos], 1usize, device).unwrap();
        let ring_logits = model
            .forward_with_cache(&tt, &pp, &mut ring_cache)
            .expect("ring decode step");
        let legacy_logits = model
            .forward_with_cache(&tt, &pp, &mut legacy_cache)
            .expect("legacy decode step");
        let ring_row: Vec<f32> = ring_logits.flatten_all().unwrap().to_vec1().unwrap();
        let legacy_row: Vec<f32> = legacy_logits.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            ring_row.iter().all(|x| x.is_finite()),
            "[{label}] non-finite ring logits at step {step}"
        );
        let ra = argmax(&ring_row);
        let la = argmax(&legacy_row);
        total += 1;
        if ra == la {
            agree += 1;
        } else {
            let denom = |r: &[f32]| r.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
            let gap_r = ring_row[ra as usize] - ring_row[la as usize];
            let rel_r = gap_r / denom(&ring_row);
            let gap_l = legacy_row[la as usize] - legacy_row[ra as usize];
            let rel_l = gap_l / denom(&legacy_row);
            eprintln!(
                "[{label}] step {step} (pos {pos}) argmax mismatch: ring {ra} legacy {la}; \
                 ring row {:.4} vs {:.4} (rel gap {rel_r:.5}); \
                 legacy row {:.4} vs {:.4} (rel gap {rel_l:.5}); \
                 near-tie bar rel<0.05 (laguna_fp8_kv convention)",
                ring_row[ra as usize],
                ring_row[la as usize],
                legacy_row[la as usize],
                legacy_row[ra as usize]
            );
        }
        next = la;
    }
    eprintln!("[{label}] argmax agreement {agree}/{total}");
    (agree, total, reference)
}

#[test]
#[ignore]
fn laguna_ring_kv_matches_legacy_across_wrap_and_compaction() {
    if std::env::var("NV_LAGUNA_TEST").is_err() {
        eprintln!("set NV_LAGUNA_TEST=1 to run");
        return;
    }
    std::env::set_var("NV_LAGUNA_M1_FLASH", "0");
    eprintln!(
        "[ring_kv] pinning NV_LAGUNA_M1_FLASH=0 for BOTH legs: the legacy cache has no M=1 GQA \
         path, so like-with-like ring-vs-legacy comparison requires the FA2 decode path on both \
         sides (keeps the 97/97 exact bar); default-ON flash coverage lives in \
         laguna_step_graph token identity and laguna_logit_diff"
    );
    let ev = LagunaEval::open().expect("laguna snapshot + prompt pack");
    eprintln!("{}", ev.describe());
    let dir = ev.dir.clone();
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
    let config = LagunaConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qconfig =
        nv_weights::QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse quant config");

    let device = Device::new_cuda(0).expect("cuda device");
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("open weights");
    let model = Laguna::from_loader_quantized(config.clone(), &weights, &qconfig, &device)
        .expect("load Laguna");

    let tokenizer = ev.tokenizer().expect("load tokenizer");
    let mut full_ids: Vec<u32> = ev
        .builder()
        .expect("pack scaffold")
        .ids_at_least(720, model.config().max_position_embeddings)
        .expect("build a templated prompt of at least 720 ids");
    full_ids.push(*full_ids.last().unwrap());

    let ids_a: Vec<u32> = full_ids[..700].to_vec();
    let (agree_a, total_a, ref_a) = lockstep_compare(&model, &device, &ids_a, 96, "wrap+compact");

    let ids_b: Vec<u32> = full_ids[..500].to_vec();
    let (agree_b, total_b, ref_b) = lockstep_compare(&model, &device, &ids_b, 96, "window-cross");

    std::env::set_var("NV_LAGUNA_RING_ATTN", "1");
    let (agree_r, total_r, ref_r) =
        lockstep_compare(&model, &device, &ids_b, 96, "ring-attn(report-only)");
    std::env::remove_var("NV_LAGUNA_RING_ATTN");
    eprintln!("ring-attn opt-in agreement {agree_r}/{total_r} (not asserted)");

    for (label, r) in [
        ("wrap+compact", &ref_a),
        ("window-cross", &ref_b),
        ("ring-attn", &ref_r),
    ] {
        assert_publishable(&ev.inspect(label, r, &tokenizer), false);
    }
    assert_eq!(
        agree_a, total_a,
        "ring vs legacy argmax disagreement across wrap+compaction"
    );
    assert_eq!(
        agree_b, total_b,
        "ring vs legacy argmax disagreement across window boundary"
    );
}
