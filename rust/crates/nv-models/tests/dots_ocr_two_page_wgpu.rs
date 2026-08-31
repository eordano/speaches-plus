#![cfg(feature = "wgpu")]

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use nv_kernels::wgpu_backend::WgpuContext;
use nv_models::deepseek_ocr::preprocess::RgbImage;
use nv_models::dots_ocr::pipeline::{
    DotsMode, DotsOcrPipeline, DotsVisionBackend, DOTS_VISION_BACKEND_ENV,
};
use nv_models::dots_ocr::preprocess::{prepare, PixelBudget};
use nv_models::dots_ocr::vision::DotsVisionConfig;
use nv_models::dots_ocr::vision_wgpu::{AttnArm, DotsVisionTowerWgpu, FlashPlan, VisionGemmRoute};
use nv_models::dots_ocr::GenerateOptions;
use nv_weights::WeightLoader;

const PAGE_ENV: &str = "NV_DOTS_PAGE";

const CONSEC_FORWARDS_ENV: &str = "NV_DOTS_CONSEC_FORWARDS";

fn consec_forwards_unset_means_two_because_two_is_the_shape_the_reclaim_defect_reproduced_on() -> usize {
    let n = std::env::var(CONSEC_FORWARDS_ENV)
        .ok()
        .filter(|v| !v.is_empty())
        .map(|v| {
            v.parse::<usize>()
                .unwrap_or_else(|_| panic!("{CONSEC_FORWARDS_ENV}={v} must be an integer >= 2"))
        })
        .unwrap_or(2);
    assert!(
        n >= 2,
        "{CONSEC_FORWARDS_ENV}={n} cannot exercise cross-forward reclaim; use >= 2"
    );
    n
}

const SECOND_FORWARD_WALL_MUST_SIT_WITHIN_25_PERCENT_OF_THE_FIRST_BECAUSE_THE_RESIDENCY_DEFECT_READS_2X_OR_LIVELOCK_WHILE_THERMAL_CLOCK_DECAY_MEASURED_1_055_ON_A_HEALTHY_RUN:
    f64 = 1.25;

const SECOND_FORWARD_MAY_RUN_UP_TO_25_PERCENT_FASTER_ON_WARM_PIPELINES_BUT_FASTER_STILL_MEANS_IT_DID_NOT_DO_THE_WORK:
    f64 = 0.75;

const PIPELINE_SMOKE_PIXEL_BUDGET_IS_SMALL_BECAUSE_THE_TWO_PAGE_SUITE_ALREADY_PAYS_FULL_RESOLUTION: usize =
    640 * 640;

fn ctx_or_fail(test: &str) -> &'static WgpuContext {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("[{test}] adapter: {}", ctx.summary());
            ctx
        }
        Err(e) => panic!(
            "{test} needs a wgpu adapter and must fail rather than skip: {e}. \
             Run through rust/scripts/nvk.sh, which wires the Vulkan loader."
        ),
    }
}

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_DOTS_DIR") {
        let p = PathBuf::from(d);
        assert!(p.is_dir(), "NV_DOTS_DIR={} is not a directory", p.display());
        return p;
    }
    let home = std::env::var("HOME").expect("HOME set");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .unwrap_or_else(|e| panic!("no dots.ocr snapshots under {}: {e}", snaps.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries
        .into_iter()
        .next()
        .expect("at least one dots.ocr snapshot")
}

fn page_path() -> PathBuf {
    std::env::var(PAGE_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| panic!("{PAGE_ENV} must name a full-resolution page image"))
}

#[test]
#[ignore]
fn two_consecutive_full_resolution_forwards_complete_with_matching_wall_clocks() {
    let ctx = ctx_or_fail("dots_two_page");
    let dir = snapshot_dir();
    let img = RgbImage::decode_file(&page_path()).expect("decode page");
    let prep = prepare(&img, PixelBudget::from_env()).expect("prepare page");
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).expect("open dots.ocr weights");
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    let tower = DotsVisionTowerWgpu::from_loader_route(
        &loader,
        "vision_tower.",
        DotsVisionConfig::default(),
        route,
    )
    .expect("wgpu tower load");
    eprintln!(
        "[dots_two_page] patches={} arm={} plan={} route={}",
        prep.grid_h * prep.grid_w,
        AttnArm::from_env().name(),
        FlashPlan::from_env().name(),
        route.name()
    );

    let n_forwards = consec_forwards_unset_means_two_because_two_is_the_shape_the_reclaim_defect_reproduced_on();
    let mut walls = Vec::new();
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for pass in 0..n_forwards {
        let t = Instant::now();
        let out = tower
            .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
            .unwrap_or_else(|e| panic!("full-resolution forward {pass} of {n_forwards} failed: {e}"));
        let wall = t.elapsed().as_secs_f64();
        assert_eq!(
            out.iter().filter(|v| !v.is_finite()).count(),
            0,
            "forward {pass} produced non-finite features"
        );
        eprintln!("[dots_two_page] pass{pass} {wall:.2}s len={}", out.len());
        walls.push(wall);
        outputs.push(out);
    }

    for pass in 1..n_forwards {
        let mismatched = outputs[0]
            .iter()
            .zip(outputs[pass].iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatched, 0,
            "forwards of the same page in one process must agree bit for bit; \
             pass{pass} differs from pass0 on {mismatched} of {} elements",
            outputs[0].len()
        );
    }

    let summary: Vec<String> = walls
        .iter()
        .enumerate()
        .map(|(i, w)| format!("pass{i} {w:.2}s"))
        .collect();
    eprintln!("[dots_two_page] {}", summary.join(" "));
    for pass in 1..n_forwards {
        let ratio = walls[pass] / walls[0];
        eprintln!("[dots_two_page] pass{pass}/pass0 ratio {ratio:.3}");
        assert!(
            ratio <= SECOND_FORWARD_WALL_MUST_SIT_WITHIN_25_PERCENT_OF_THE_FIRST_BECAUSE_THE_RESIDENCY_DEFECT_READS_2X_OR_LIVELOCK_WHILE_THERMAL_CLOCK_DECAY_MEASURED_1_055_ON_A_HEALTHY_RUN,
            "forward {pass} took {:.2}s against pass0's {:.2}s ({ratio:.3}x): \
             the tower's end-of-forward blocking reclaim (VisionGemm::reclaim_dropped_device_buffers) \
             is not freeing the prior page's dropped buffers before the next page runs; the opt-in \
             NV_WGPU_RECLAIM_POLL dispatch-side poll is a separate, unmeasured knob and is NOT what \
             this gate exercises",
            walls[pass],
            walls[0]
        );
        assert!(
            ratio >= SECOND_FORWARD_MAY_RUN_UP_TO_25_PERCENT_FASTER_ON_WARM_PIPELINES_BUT_FASTER_STILL_MEANS_IT_DID_NOT_DO_THE_WORK,
            "forward {pass} took only {:.2}s against pass0's {:.2}s ({ratio:.3}x); \
             a pass that cheap did not run the same graph",
            walls[pass],
            walls[0]
        );
    }
}

#[test]
#[ignore]
fn the_env_selected_wgpu_tower_feeds_the_unchanged_decoder_end_to_end() {
    let _ = ctx_or_fail("dots_pipeline_wgpu");
    let dir = snapshot_dir();
    let img = RgbImage::decode_file(&page_path()).expect("decode page");
    let device = Device::new_cuda(0).expect("the dots.ocr decoder path under test runs on cuda");

    std::env::set_var(DOTS_VISION_BACKEND_ENV, "wgpu");
    let loaded = DotsOcrPipeline::load(&dir, &device);
    std::env::remove_var(DOTS_VISION_BACKEND_ENV);
    let mut pipeline = loaded.expect("pipeline load with the wgpu vision backend");

    assert_eq!(pipeline.vision_backend(), DotsVisionBackend::Wgpu);
    assert!(
        pipeline.vision().is_none(),
        "the wgpu backend must not have built the candle tower"
    );
    pipeline.set_pixel_budget(PixelBudget {
        min_pixels: PixelBudget::default().min_pixels,
        max_pixels: PIPELINE_SMOKE_PIXEL_BUDGET_IS_SMALL_BECAUSE_THE_TWO_PAGE_SUITE_ALREADY_PAYS_FULL_RESOLUTION,
    });

    let prep = pipeline.prepare(&img).expect("prepare page");
    let feats = pipeline.encode_vision(&prep).expect("wgpu vision encode");
    assert_eq!(
        feats.dims(),
        [
            prep.num_vision_tokens(),
            DotsVisionConfig::default().hidden_size
        ],
        "wgpu vision features must arrive shaped for the decoder"
    );

    let opts = GenerateOptions {
        max_new_tokens: 24,
        ..GenerateOptions::default()
    };
    let out = pipeline
        .recognize(&img, DotsMode::PlainOcr, &opts)
        .expect("recognize through the wgpu tower and the unchanged decoder");
    eprintln!(
        "[dots_pipeline_wgpu] grid={:?} prompt_tokens={} generated={} raw head: {}",
        out.grid,
        out.prompt_tokens,
        out.generated_tokens,
        out.raw.chars().take(80).collect::<String>()
    );
    assert!(
        out.generated_tokens > 0,
        "the decoder generated nothing from wgpu vision features"
    );
}

#[test]
fn unset_env_defers_to_the_device_and_only_metal_defaults_to_the_wgpu_tower() {
    let cpu = Device::Cpu;
    std::env::remove_var(DOTS_VISION_BACKEND_ENV);
    assert_eq!(
        DotsVisionBackend::from_env_for(&cpu).unwrap(),
        DotsVisionBackend::Candle,
        "the wgpu tower was measured on metal only, so no other device may inherit it unasked"
    );
    assert_eq!(
        DotsVisionBackend::default_for(&cpu),
        DotsVisionBackend::Candle
    );
    if let Ok(metal) = Device::new_metal(0) {
        assert_eq!(
            DotsVisionBackend::default_for(&metal),
            DotsVisionBackend::Wgpu,
            "metal is the adapter the 1.78 s/page tower win was measured on"
        );
    }
    std::env::set_var(DOTS_VISION_BACKEND_ENV, "0");
    assert_eq!(
        DotsVisionBackend::from_env_for(&cpu).unwrap(),
        DotsVisionBackend::Candle,
        "explicit =0 must restore the prior default"
    );
    std::env::set_var(DOTS_VISION_BACKEND_ENV, "wgpu");
    assert_eq!(
        DotsVisionBackend::from_env_for(&cpu).unwrap(),
        DotsVisionBackend::Wgpu,
        "an explicitly named arm must be honoured on any device"
    );
    std::env::remove_var(DOTS_VISION_BACKEND_ENV);
}
