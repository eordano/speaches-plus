#![cfg(any(feature = "cuda", feature = "metal"))]

mod hub_snapshot;

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use nv_models::deepseek_ocr::preprocess::RgbImage;
use nv_models::dots_ocr::{DotsMode, DotsOcrPipeline, GenerateOptions};

const PAGE_ENV: &str = "NV_DOTS_PAGE";
const OUT_ENV: &str = "NV_DOTS_PAGE_OUT";

const MAX_NEW_TOKENS_4096_BECAUSE_A_DENSE_HANDWRITTEN_NOTEBOOK_PAGE_OVERRUNS_THE_2048_BENCH_CAP:
    usize = 4096;

fn serving_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    Device::Cpu
}

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
fn dots_ocr_one_page_wall_clock_and_text_for_an_operator_supplied_image() {
    let Ok(page) = std::env::var(PAGE_ENV) else {
        hub_snapshot::precondition_absent(
            "dots_ocr_one_page_wall_clock_and_text_for_an_operator_supplied_image",
            "NV_DOTS_PAGE unset",
            "set NV_DOTS_PAGE to an image path; NV_DOTS_PAGE_OUT optionally names a file for the recognized text",
        );
        return;
    };
    let page = PathBuf::from(page);
    let dir = snapshot_dir().expect("dots.ocr snapshot present");
    let revision = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let img = RgbImage::decode_file(&page).expect("decode operator image");
    let (w, h) = (img.w, img.h);

    let device = serving_device();
    let load_t0 = Instant::now();
    let pipe = DotsOcrPipeline::load(&dir, &device).expect("load");
    let load_s = load_t0.elapsed().as_secs_f64();
    let precision = pipe.precision();
    let budget = pipe.pixel_budget();

    let prep_t0 = Instant::now();
    let prep = pipe.prepare(&img).expect("prepare");
    let prep_s = prep_t0.elapsed().as_secs_f64();
    let n_vis = prep.num_vision_tokens();

    let opts = GenerateOptions {
        max_new_tokens:
            MAX_NEW_TOKENS_4096_BECAUSE_A_DENSE_HANDWRITTEN_NOTEBOOK_PAGE_OVERRUNS_THE_2048_BENCH_CAP,
        ..GenerateOptions::default()
    };

    let t0 = Instant::now();
    let res = pipe
        .recognize(&img, DotsMode::PlainOcr, &opts)
        .expect("recognize");
    let wall_s = t0.elapsed().as_secs_f64();

    let tok_s = res.generated_tokens as f64 / wall_s.max(1e-6);
    eprintln!(
        "DOTS-PAGE file={} src_px={w}x{h} vision_tokens={n_vis} prompt_tokens={} generated={} \
         load_s={load_s:.2} prepare_s={prep_s:.3} recognize_wall_s={wall_s:.3} decode_tok_s={tok_s:.2} \
         precision={precision:?} budget={budget:?} mode=PlainOcr max_new_tokens={} \
         basis=whole_recognize_wall_incl_vision_and_prefill_excl_load \
         checkpoint=rednote-hilab/dots.ocr@{revision}",
        page.display(),
        res.prompt_tokens,
        res.generated_tokens,
        opts.max_new_tokens,
    );

    if let Ok(out) = std::env::var(OUT_ENV) {
        std::fs::write(&out, res.text.as_bytes()).expect("write recognized text");
        eprintln!("DOTS-PAGE wrote {} bytes to {out}", res.text.len());
    }
    eprintln!("---- BEGIN TEXT ----\n{}\n---- END TEXT ----", res.text);

    assert!(
        res.generated_tokens > 0,
        "generated zero tokens for {}",
        page.display()
    );
}
