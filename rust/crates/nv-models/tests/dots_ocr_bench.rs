#![cfg(feature = "cuda")]

mod common;
use common::cer;
use common::fixture_expected_text;
use common::fixture_rgb;
use common::FIXTURES;
use common::wer;
mod hub_snapshot;

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use nv_models::dots_ocr::{DotsMode, DotsOcrPipeline, GenerateOptions, RgbImage};

fn snapshot_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_DOTS_DIR") {
        let p = PathBuf::from(d);
        return p.is_dir().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries.into_iter().next()
}

#[test]
#[ignore]
fn dots_ocr_baseline_speed_and_cer_over_070_071_fixtures() {
    if std::env::var("NV_DOTS_BENCH").as_deref() != Ok("1") {
        hub_snapshot::precondition_absent(
            "dots_ocr_baseline_speed_and_cer_over_070_071_fixtures",
            "NV_DOTS_BENCH != 1",
            "set NV_DOTS_BENCH=1 and NV_DOTS_DIR to a rednote-hilab/dots.ocr snapshot (or leave unset to use the hub cache)",
        );
        return;
    }
    let dir = snapshot_dir().expect("dots.ocr snapshot present");
    let device = Device::new_cuda(0).expect("cuda device 0");
    let load_t0 = Instant::now();
    let pipe = DotsOcrPipeline::load(&dir, &device).expect("load");
    eprintln!("[dots-bench] load elapsed {:?}", load_t0.elapsed());

    let opts = GenerateOptions {
        max_new_tokens: 2048,
        ..GenerateOptions::default()
    };

    let mut cers: Vec<f64> = Vec::new();
    let mut wers: Vec<f64> = Vec::new();
    let mut timed_secs: Vec<f64> = Vec::new();
    let mut timed_tokens: Vec<usize> = Vec::new();

    for (i, name) in FIXTURES.iter().enumerate() {
        let img = fixture_rgb(name);
        let want = fixture_expected_text(name);
        let t0 = Instant::now();
        let res = pipe
            .recognize(&img, DotsMode::PlainOcr, &opts)
            .expect("recognize");
        let elapsed = t0.elapsed().as_secs_f64();
        let e_cer = cer(&res.text, &want);
        let e_wer = wer(&res.text, &want);
        eprintln!(
            "[dots-bench] {name}: {:.3}s, {} generated tokens, {:.1} tok/s, CER={e_cer:.4} WER={e_wer:.4}",
            elapsed,
            res.generated_tokens,
            res.generated_tokens as f64 / elapsed.max(1e-6),
        );
        cers.push(e_cer);
        wers.push(e_wer);
        if i > 0 {
            timed_secs.push(elapsed);
            timed_tokens.push(res.generated_tokens);
        }
    }

    let n = cers.len();
    let mean_cer = cers.iter().sum::<f64>() / n as f64;
    let mean_wer = wers.iter().sum::<f64>() / n as f64;
    let total_secs: f64 = timed_secs.iter().sum();
    let total_tokens: usize = timed_tokens.iter().sum();
    let mean_ms_per_image = total_secs * 1e3 / timed_secs.len() as f64;
    let pages_per_sec = timed_secs.len() as f64 / total_secs;
    eprintln!(
        "[dots-bench] SUMMARY basis=rednote-hilab/dots.ocr snapshot={} device=cuda bf16 mode=PlainOcr prompt=\"Extract the text content from this image.\" max_new_tokens={} images={n} (1 warmup discarded from timing) \
         mean_ms_per_image={mean_ms_per_image:.1} pages_per_sec={pages_per_sec:.3} total_tokens={total_tokens} tok_per_s={:.1} mean_CER={mean_cer:.4} mean_WER={mean_wer:.4}",
        dir.display(),
        opts.max_new_tokens,
        total_tokens as f64 / total_secs,
    );

    assert_eq!(cers.len(), FIXTURES.len(), "must score every fixture image");
    assert!(total_tokens > 0, "generated zero tokens across the timed set");
}
