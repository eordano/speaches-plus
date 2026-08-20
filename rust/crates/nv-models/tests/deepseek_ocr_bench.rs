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
use nv_models::deepseek_ocr::{
    DecoderPrecision, DeepSeekOcr2Pipeline, GenerateOptions, ResolutionMode, RgbImage,
    PROMPT_FREE_OCR,
};
use common::deepseek_ocr_snapshot_dir as snapshot_dir;

#[test]
#[ignore]
fn deepseek_markdown_truncation_census_over_070_071_fixtures() {
    use nv_models::deepseek_ocr::decoder::strip_grounding_tokens;
    use nv_models::deepseek_ocr::preprocess::prepare;
    use nv_models::deepseek_ocr::{build_prompt_tokens, PROMPT_GROUNDING_MARKDOWN};

    if std::env::var("NV_DSOCR_BENCH").as_deref() != Ok("1") {
        hub_snapshot::precondition_absent(
            "deepseek_markdown_truncation_census_over_070_071_fixtures",
            "NV_DSOCR_BENCH != 1",
            "set NV_DSOCR_BENCH=1; the deepseek-ai/DeepSeek-OCR-2 checkpoint IS cached on this box, so this is an opt-in knob, not a missing artifact",
        );
        return;
    }
    let dir = snapshot_dir().expect("DeepSeek-OCR-2 snapshot present");
    let device = Device::new_cuda(0).expect("cuda device 0");
    let pipe = DeepSeekOcr2Pipeline::load(&dir, &device, DecoderPrecision::Bf16).expect("load");
    let opts = GenerateOptions {
        max_new_tokens: 4096,
        ..GenerateOptions::recipe()
    };

    let mut capped: Vec<&str> = Vec::new();
    let mut raw_counts: Vec<usize> = Vec::new();
    for name in FIXTURES {
        let img = fixture_rgb(name);
        let prep = prepare(&img, ResolutionMode::Gundam).expect("prepare");
        let feats = pipe.vision().encode_prepared(&prep).expect("vision encode");
        let tokens = build_prompt_tokens(
            |s| pipe.encode_text(s),
            PROMPT_GROUNDING_MARKDOWN,
            prep.vision_tokens(),
        )
        .expect("prompt tokens");
        let t0 = Instant::now();
        let out = pipe
            .decoder()
            .generate_detected(&tokens, Some(&feats), &opts)
            .expect("generate_detected");
        let visible = strip_grounding_tokens(&out.tokens);
        let looped = out.loop_detection.is_some();
        let overhead = out.tokens.len() as f64 / visible.len().max(1) as f64;
        eprintln!(
            "[md-census] {name}: raw={} visible={} grounding_overhead={overhead:.2}x \
             hit_eos={} looped={looped} {:.2}s",
            out.tokens.len(),
            visible.len(),
            out.hit_eos,
            t0.elapsed().as_secs_f64(),
        );
        raw_counts.push(out.tokens.len());
        if !out.hit_eos && !looped {
            capped.push(name);
        }
    }
    raw_counts.sort_unstable();
    eprintln!(
        "[md-census] SUMMARY basis=deepseek-ai/DeepSeek-OCR-2 device=cuda bf16 mode=Gundam \
         prompt=grounding_markdown max_new_tokens={} images={} capped={:?} \
         raw_tokens median={} max={}",
        opts.max_new_tokens,
        FIXTURES.len(),
        capped,
        raw_counts[raw_counts.len() / 2],
        raw_counts.last().unwrap(),
    );
    assert_eq!(raw_counts.len(), FIXTURES.len(), "must census every fixture image");
}

#[test]
#[ignore]
fn deepseek_ocr2_baseline_speed_and_cer_over_070_071_fixtures() {
    if std::env::var("NV_DSOCR_BENCH").as_deref() != Ok("1") {
        hub_snapshot::precondition_absent(
            "deepseek_ocr2_baseline_speed_and_cer_over_070_071_fixtures",
            "NV_DSOCR_BENCH != 1",
            "set NV_DSOCR_BENCH=1; the deepseek-ai/DeepSeek-OCR-2 checkpoint IS cached on this box, so this is an opt-in knob, not a missing artifact",
        );
        return;
    }
    let dir = snapshot_dir().expect("DeepSeek-OCR-2 snapshot present");
    let device = Device::new_cuda(0).expect("cuda device 0");
    let load_t0 = Instant::now();
    let pipe = DeepSeekOcr2Pipeline::load(&dir, &device, DecoderPrecision::Bf16).expect("load");
    eprintln!("[dsocr-bench] load elapsed {:?}", load_t0.elapsed());

    let opts = GenerateOptions {
        max_new_tokens: 2048,
        ..GenerateOptions::recipe()
    };

    let mut cers: Vec<f64> = Vec::new();
    let mut wers: Vec<f64> = Vec::new();
    let mut timed_secs: Vec<f64> = Vec::new();
    let mut timed_tokens: Vec<usize> = Vec::new();

    for (i, name) in FIXTURES.iter().enumerate() {
        let img = fixture_rgb(name);
        let want = fixture_expected_text(name);
        let t0 = Instant::now();
        let (tokens, _prep) = pipe
            .generate_tokens(&img, PROMPT_FREE_OCR, ResolutionMode::Gundam, &opts)
            .expect("generate_tokens");
        let elapsed = t0.elapsed().as_secs_f64();
        let text = pipe
            .tokenizer()
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))
            .expect("decode");
        let e_cer = cer(&text, &want);
        let e_wer = wer(&text, &want);
        eprintln!(
            "[dsocr-bench] {name}: {:.3}s, {} tokens, {:.1} tok/s, CER={e_cer:.4} WER={e_wer:.4}",
            elapsed,
            tokens.len(),
            tokens.len() as f64 / elapsed.max(1e-6),
        );
        cers.push(e_cer);
        wers.push(e_wer);
        if i > 0 {
            timed_secs.push(elapsed);
            timed_tokens.push(tokens.len());
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
        "[dsocr-bench] SUMMARY basis=deepseek-ai/DeepSeek-OCR-2 snapshot={} device=cuda bf16 mode=Gundam prompt=free_ocr max_new_tokens={} images={n} (1 warmup discarded from timing) \
         mean_ms_per_image={mean_ms_per_image:.1} pages_per_sec={pages_per_sec:.3} total_tokens={total_tokens} tok_per_s={:.1} mean_CER={mean_cer:.4} mean_WER={mean_wer:.4}",
        dir.display(),
        opts.max_new_tokens,
        total_tokens as f64 / total_secs,
    );

    assert_eq!(cers.len(), FIXTURES.len(), "must score every fixture image");
    assert!(total_tokens > 0, "generated zero tokens across the timed set");
}
