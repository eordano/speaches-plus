#![cfg(feature = "cuda")]

mod common;
use common::fixture_rgb;
mod hub_snapshot;

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{Device, Tensor};
use nv_models::deepseek_ocr::decoder_graph_batch::{BatchSampler, DsocrBatchDecodeGraph};
use nv_models::deepseek_ocr::{
    DecoderPrecision, DeepSeekOcr2Pipeline, DsocrDecodeGraph, GenerateOptions, ResolutionMode,
    RgbImage, PROMPT_FREE_OCR,
};
use common::deepseek_ocr_snapshot_dir as snapshot_dir;

const PAGES: &[&str] = &[
    "071-ocr-layout-invoice",
    "071-ocr-layout-newspaper",
    "071-ocr-layout-report",
    "071-ocr-layout-letter",
    "071-ocr-layout-labnotes",
];

#[test]
fn bucket_selection_rounds_up_and_rejects_zero() {
    use nv_models::deepseek_ocr::decoder_graph_batch::{bucket_for, parse_buckets};
    let b = parse_buckets(Some("8,1,4,2,4"));
    assert_eq!(b, vec![1, 2, 4, 8]);
    assert_eq!(bucket_for(&b, 0), None);
    assert_eq!(bucket_for(&b, 1), Some(1));
    assert_eq!(bucket_for(&b, 3), Some(4));
    assert_eq!(bucket_for(&b, 8), Some(8));
    assert_eq!(bucket_for(&b, 9), None);
    assert_eq!(parse_buckets(None), vec![1, 2, 4, 8]);
    assert_eq!(parse_buckets(Some("junk")), vec![1, 2, 4, 8]);
}

struct Page {
    tokens: Vec<u32>,
    feats: Tensor,
}

#[test]
#[ignore]
fn batched_decode_matches_single_sequence_tokens() {
    if std::env::var("LOAD_DSOCR").as_deref() != Ok("1") {
        hub_snapshot::precondition_absent(
            "deepseek_ocr_graph_batch (bs=N gate)",
            "LOAD_DSOCR != 1",
            "set LOAD_DSOCR=1; the deepseek-ai/DeepSeek-OCR-2 checkpoint IS cached on this box, so this is an opt-in knob, not a missing artifact",
        );
        return;
    }
    std::env::set_var("NV_DSOCR_DECODE", "kernel");
    let max_new: usize = std::env::var("NV_DSOCR_BSN_MAX_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(192);
    let batch: usize = std::env::var("NV_DSOCR_BSN_B")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    let dir = snapshot_dir().expect("DeepSeek-OCR-2 snapshot present");
    let device = Device::new_cuda(0).expect("cuda device");
    let pipe = DeepSeekOcr2Pipeline::load(&dir, &device, DecoderPrecision::Bf16).expect("load");
    let decoder = pipe.decoder_arc();
    let cfg_max = decoder.config().max_position_embeddings;
    let eos = decoder.config().eos_token_id;
    let vocab = decoder.config().vocab_size;

    let mut pages: Vec<Page> = Vec::new();
    for name in PAGES.iter().take(batch) {
        let img = fixture_rgb(name);
        let prep = nv_models::deepseek_ocr::preprocess::prepare(&img, ResolutionMode::Gundam)
            .expect("prepare");
        let feats = pipe.vision().encode_prepared(&prep).expect("vision encode");
        let tokens = nv_models::deepseek_ocr::build_prompt_tokens(
            |s| pipe.encode_text(s),
            PROMPT_FREE_OCR,
            prep.vision_tokens(),
        )
        .expect("prompt tokens");
        pages.push(Page { tokens, feats });
    }

    let opts = GenerateOptions {
        max_new_tokens: max_new,
        ..GenerateOptions::recipe()
    };

    let mut reference: Vec<Vec<u32>> = Vec::new();
    {
        let mut g = DsocrDecodeGraph::new(decoder.clone(), cfg_max).expect("bs=1 graph");
        let t = Instant::now();
        for p in &pages {
            let out = g
                .generate(&p.tokens, Some(&p.feats), &opts)
                .expect("bs=1 generate");
            reference.push(out.tokens);
        }
        let secs = t.elapsed().as_secs_f64();
        let n: usize = reference.iter().map(|r| r.len()).sum();
        eprintln!(
            "[bsn-gate] bs=1 reference: {n} tokens in {secs:.3}s = {:.1} tok/s",
            n as f64 / secs
        );
    }

    let mut graph =
        DsocrBatchDecodeGraph::new(decoder.clone(), cfg_max, vec![1, 2, 4, 8]).expect("bs=N graph");
    let b = graph.bucket_for(pages.len()).expect("bucket exists");
    let mut all: Vec<Vec<u32>> = Vec::new();
    let mut gen: Vec<Vec<u32>> = Vec::new();
    let mut rngs: Vec<BatchSampler> = Vec::new();
    let mut next: Vec<Option<u32>> = vec![None; b];
    let mut live = vec![false; b];

    for (j, p) in pages.iter().enumerate() {
        let max_len = (p.tokens.len() + max_new).min(cfg_max);
        let mut logits = graph
            .prefill_slot(j, &p.tokens, Some(&p.feats), max_len)
            .expect("prefill slot");
        let mut rng = BatchSampler::new(opts.seed);
        let mut toks = p.tokens.clone();
        let first = rng
            .next_token(&mut logits, &toks, &opts)
            .expect("first token");
        toks.push(first);
        all.push(toks);
        gen.push(vec![first]);
        rngs.push(rng);
        next[j] = Some(first);
        live[j] = first != eos;
        if !live[j] {
            graph.release_slot(j);
            next[j] = None;
        }
    }

    let t = Instant::now();
    let mut steps = 0usize;
    let mut nodes_after_capture = 0usize;
    while live.iter().any(|&l| l) {
        let toks: Vec<Option<u32>> = (0..b)
            .map(|j| if live[j] { next[j] } else { None })
            .collect();
        graph.step_batch(&toks).expect("step_batch");
        if steps == 0 {
            nodes_after_capture = graph.node_count();
        }
        let logits = graph.logits_batch(b).expect("logits_batch");
        steps += 1;
        for j in 0..b {
            if !live[j] {
                continue;
            }
            let mut row = logits[j * vocab..(j + 1) * vocab].to_vec();
            let nt = rngs[j]
                .next_token(&mut row, &all[j], &opts)
                .expect("next token");
            gen[j].push(nt);
            all[j].push(nt);
            next[j] = Some(nt);
            if nt == eos || gen[j].len() >= max_new {
                live[j] = false;
                graph.release_slot(j);
            }
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let n: usize = gen.iter().map(|g| g.len()).sum();
    eprintln!(
        "[bsn-gate] bs={} batched: {n} tokens in {secs:.3}s = {:.1} tok/s over {steps} steps \
         ({:.3} ms/step)",
        pages.len(),
        n as f64 / secs,
        secs * 1e3 / steps as f64
    );
    if steps > 0 {
        let nodes_after_replays = graph.node_count();
        eprintln!(
            "[bsn-gate] bucket {b}: node_count after capture {nodes_after_capture}, after \
             {steps} steps {nodes_after_replays} (prealloc={})",
            std::env::var("NV_DSOCR_GRAPH_PREALLOC").as_deref() == Ok("1"),
        );
        assert!(
            nodes_after_capture > 0,
            "captured bs=N graph reports zero nodes"
        );
        assert_eq!(
            nodes_after_capture, nodes_after_replays,
            "bs=N graph node count changed across replays"
        );
    }

    let mut mismatches = 0usize;
    for (j, name) in PAGES.iter().take(pages.len()).enumerate() {
        let r = &reference[j];
        let g = &gen[j];
        let common = r.len().min(g.len());
        let first_div = (0..common).find(|&i| r[i] != g[i]);
        if r == g {
            eprintln!("[bsn-gate] slot {j} {name}: EXACT ({} tokens)", r.len());
        } else {
            mismatches += 1;
            eprintln!(
                "[bsn-gate] slot {j} {name}: DIVERGE ref_len={} bsn_len={} first_diff={:?}",
                r.len(),
                g.len(),
                first_div
            );
        }
    }
    assert_eq!(
        mismatches,
        0,
        "bs=N token streams diverged from the bs=1 reference on {mismatches} of {} pages",
        pages.len()
    );
}
