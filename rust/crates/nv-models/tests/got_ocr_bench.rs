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
use nv_models::deepseek_ocr::RgbImage;
use nv_models::got_ocr::pipeline::{GotMode, GotOcrPipeline};
use common::got_ocr_snapshot_dir as snapshot_dir;

const GOT_BENCH_MAX_NEW_TOKENS: usize = 1024;

#[test]
#[ignore]
fn got_ocr_baseline_speed_and_cer_over_070_071_fixtures() {
    if std::env::var("NV_GOT_OCR_BENCH").as_deref() != Ok("1") {
        hub_snapshot::precondition_absent(
            "got_ocr_baseline_speed_and_cer_over_070_071_fixtures",
            "NV_GOT_OCR_BENCH != 1",
            "set NV_GOT_OCR_BENCH=1 and NV_GOT_OCR_DIR to a stepfun-ai/GOT-OCR-2.0-hf snapshot (or leave unset to use the hub cache)",
        );
        return;
    }
    let dir = snapshot_dir().expect("GOT-OCR-2.0-hf snapshot present");
    let device = Device::new_cuda(0).expect("cuda device 0");
    let load_t0 = Instant::now();
    let pipe = GotOcrPipeline::load(&dir, &device).expect("load");
    eprintln!("[got-bench] load elapsed {:?}", load_t0.elapsed());

    let mut cers: Vec<f64> = Vec::new();
    let mut wers: Vec<f64> = Vec::new();
    let mut timed_secs: Vec<f64> = Vec::new();
    let mut timed_tokens: Vec<usize> = Vec::new();

    let mut encode_secs: Vec<f64> = Vec::new();
    for (i, name) in FIXTURES.iter().enumerate() {
        let img = fixture_rgb(name);
        let want = fixture_expected_text(name);
        let te = Instant::now();
        let _feats = pipe.encode_image(&img).expect("encode");
        let enc = te.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let res = pipe
            .recognize(&img, GotMode::Plain, GOT_BENCH_MAX_NEW_TOKENS)
            .expect("recognize");
        let elapsed = t0.elapsed().as_secs_f64();
        let e_cer = cer(&res.text, &want);
        let e_wer = wer(&res.text, &want);
        eprintln!(
            "[got-bench] {name}: {:.3}s (encode {:.3}s), {} generated tokens, {:.1} tok/s, CER={e_cer:.4} WER={e_wer:.4}",
            elapsed,
            enc,
            res.generated_tokens,
            res.generated_tokens as f64 / elapsed.max(1e-6),
        );
        if i > 0 {
            encode_secs.push(enc);
        }
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
    let mean_encode_ms = encode_secs.iter().sum::<f64>() * 1e3 / encode_secs.len().max(1) as f64;
    let pages_per_sec = timed_secs.len() as f64 / total_secs;
    eprintln!(
        "[got-bench] SPLIT mean_encode_ms={mean_encode_ms:.1} mean_total_ms={mean_ms_per_image:.1} encode_share={:.1}%",
        100.0 * mean_encode_ms / mean_ms_per_image,
    );
    eprintln!(
        "[got-bench] SUMMARY basis=stepfun-ai/GOT-OCR-2.0-hf snapshot={} device=cuda bf16 mode=Plain prompt=\"OCR: \" max_new_tokens={GOT_BENCH_MAX_NEW_TOKENS} images={n} (1 warmup discarded from timing) \
         mean_ms_per_image={mean_ms_per_image:.1} pages_per_sec={pages_per_sec:.3} total_tokens={total_tokens} tok_per_s={:.1} mean_CER={mean_cer:.4} mean_WER={mean_wer:.4}",
        dir.display(),
        total_tokens as f64 / total_secs,
    );

    assert_eq!(cers.len(), FIXTURES.len(), "must score every fixture image");
    assert!(total_tokens > 0, "generated zero tokens across the timed set");
}
