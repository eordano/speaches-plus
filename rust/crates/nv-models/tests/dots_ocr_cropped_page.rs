#![cfg(any(feature = "cuda", feature = "metal"))]

mod hub_snapshot;

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use nv_models::deepseek_ocr::preprocess::RgbImage;
use nv_models::dots_ocr::{DotsOcrPipeline, GenerateOptions};

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

#[test]
#[ignore]
fn dots_ocr_cropped_page_layout_detect_then_per_region_ocr_for_an_operator_supplied_image() {
    let Ok(page) = std::env::var(PAGE_ENV) else {
        hub_snapshot::precondition_absent(
            "dots_ocr_cropped_page_layout_detect_then_per_region_ocr_for_an_operator_supplied_image",
            "NV_DOTS_PAGE unset",
            "set NV_DOTS_PAGE to an image path; NV_DOTS_PAGE_OUT optionally names a file for the stitched text",
        );
        return;
    };
    let page = PathBuf::from(page);
    let dir = hub_snapshot::dir_from_env_or_hub(
        "NV_DOTS_DIR",
        "rednote-hilab/dots.ocr",
        &["config.json", "tokenizer.json", "*.safetensors"],
    )
    .expect("dots.ocr snapshot present");
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

    let opts = GenerateOptions {
        max_new_tokens:
            MAX_NEW_TOKENS_4096_BECAUSE_A_DENSE_HANDWRITTEN_NOTEBOOK_PAGE_OVERRUNS_THE_2048_BENCH_CAP,
        ..GenerateOptions::default()
    };

    let t0 = Instant::now();
    let res = pipe
        .recognize_cropped(&img, &opts)
        .expect("recognize_cropped");
    let wall_s = t0.elapsed().as_secs_f64();

    eprintln!(
        "DOTS-CROPPED file={} src_px={w}x{h} regions={} prompt_tokens={} generated={} \
         load_s={load_s:.2} wall_s={wall_s:.3} looped={} hit_eos={} truncated={} \
         precision={precision:?} layout_max={} crop_max={} max_new_tokens={} \
         basis=layout_pass_plus_all_crop_recognize_walls_excl_load \
         checkpoint=rednote-hilab/dots.ocr@{revision}",
        page.display(),
        res.page.elements.len(),
        res.prompt_tokens,
        res.generated_tokens,
        res.looped,
        res.hit_eos,
        res.page.truncated,
        std::env::var("NV_DOTS_LAYOUT_MAX_PIXELS").unwrap_or_else(|_| "default".into()),
        std::env::var("NV_DOTS_CROP_MAX_PIXELS").unwrap_or_else(|_| "default".into()),
        opts.max_new_tokens,
    );
    for (i, e) in res.page.elements.iter().enumerate() {
        eprintln!(
            "DOTS-CROPPED region[{i}] bbox={:?} category={:?} chars={}",
            e.bbox,
            e.category,
            e.text.as_deref().map(str::len).unwrap_or(0)
        );
    }

    eprintln!(
        "---- LAYOUT RAW ----\n{}\n---- END LAYOUT RAW ----",
        res.raw
    );
    if let Ok(out) = std::env::var(OUT_ENV) {
        std::fs::write(&out, res.text.as_bytes()).expect("write stitched text");
        eprintln!("DOTS-CROPPED wrote {} bytes to {out}", res.text.len());
    }
    eprintln!("---- BEGIN TEXT ----\n{}\n---- END TEXT ----", res.text);

    assert!(
        res.generated_tokens > 0,
        "generated zero tokens for {}",
        page.display()
    );
    assert!(
        !res.text.trim().is_empty(),
        "cropped OCR stitched empty text for {}",
        page.display()
    );
}
