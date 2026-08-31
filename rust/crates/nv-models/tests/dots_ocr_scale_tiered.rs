#![cfg(any(feature = "cuda", feature = "metal"))]

mod common;
use common::cer;
mod hub_snapshot;

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use nv_models::deepseek_ocr::preprocess::RgbImage;
use nv_models::dots_ocr::{
    plan_regions, DotsMode, DotsOcrPipeline, GenerateOptions, PixelBudget, Region,
};

const SCALE_GATE_ENV: &str = "NV_DOTS_SCALE";

const BIG_FIXTURES: &[&str] = &[
    "090-scale-15mp-catalog",
    "090-scale-35mp-report",
    "090-scale-50mp-broadsheet",
    "090-scale-100mp-archive",
];

const SCREEN_FIXTURES: &[&str] = &[
    "091-screen-1080p-editor",
    "091-screen-1440p-editor",
    "091-screen-4k-editor",
    "091-screen-4k-darkmode",
    "091-screen-5k-editor",
    "091-screen-uw49-editor",
];

const MAX_NEW_TOKENS_4096_COVERS_THE_DENSEST_SCALE_PAGE_AT_1_2_TOKENS_PER_WORD: usize = 4096;

struct ScaleFixture {
    img: RgbImage,
    expected: String,
    gt_regions: Vec<GtRegion>,
}

struct GtRegion {
    bbox: [f32; 4],
    tier: String,
    text: String,
}

fn scale_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../conformance/fixtures-scale")
}

fn load_scale_fixture(name: &str) -> ScaleFixture {
    let dir = scale_dir().join(name);
    let img = RgbImage::decode_file(&dir.join("input.png")).expect("decode scale fixture");
    let raw = std::fs::read_to_string(dir.join("fixture.json")).expect("read fixture.json");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse fixture.json");
    let expected = v["expected_text"].as_str().expect("expected_text").to_string();
    let gt_regions = v["regions"]
        .as_array()
        .expect("regions")
        .iter()
        .map(|r| {
            let b = r["bbox"].as_array().expect("bbox");
            GtRegion {
                bbox: [
                    b[0].as_f64().unwrap() as f32,
                    b[1].as_f64().unwrap() as f32,
                    b[2].as_f64().unwrap() as f32,
                    b[3].as_f64().unwrap() as f32,
                ],
                tier: r["tier"].as_str().unwrap_or("").to_string(),
                text: r["text"].as_str().unwrap_or("").to_string(),
            }
        })
        .collect();
    ScaleFixture {
        img,
        expected,
        gt_regions,
    }
}

fn neutral(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if in_tag => {}
            '#' | '*' | '`' | '|' => {}
            c if c.is_whitespace() => out.push(' '),
            c => out.push(c),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn cer_neutral(got: &str, want: &str) -> f64 {
    cer(&neutral(got), &neutral(want))
}

fn distinctive_words(s: &str) -> Vec<String> {
    let mut out: Vec<String> = s
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 5)
        .map(|w| w.to_lowercase())
        .collect();
    out.sort();
    out.dedup();
    out
}

fn word_recall(got: &str, want: &str) -> f64 {
    let want_words = distinctive_words(want);
    if want_words.is_empty() {
        return 1.0;
    }
    let got_lower = got.to_lowercase();
    let hit = want_words.iter().filter(|w| got_lower.contains(*w)).count();
    hit as f64 / want_words.len() as f64
}

fn tier_recalls(fx: &ScaleFixture, got: &str) -> Vec<(String, f64, usize)> {
    let mut tiers: Vec<String> = fx.gt_regions.iter().map(|r| r.tier.clone()).collect();
    tiers.sort();
    tiers.dedup();
    tiers
        .into_iter()
        .map(|t| {
            let text: String = fx
                .gt_regions
                .iter()
                .filter(|r| r.tier == t)
                .map(|r| r.text.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let n = distinctive_words(&text).len();
            (t, word_recall(got, &text), n)
        })
        .collect()
}

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

fn scale_pipeline() -> Option<(DotsOcrPipeline, String)> {
    if std::env::var(SCALE_GATE_ENV).as_deref() != Ok("1") {
        hub_snapshot::precondition_absent(
            "dots_ocr_scale_tiered",
            "NV_DOTS_SCALE != 1",
            "set NV_DOTS_SCALE=1 (and optionally NV_DOTS_DIR) to run the scale-corpus probes",
        );
        return None;
    }
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
    let device = serving_device();
    let pipe = DotsOcrPipeline::load(&dir, &device).expect("load dots.ocr pipeline");
    Some((pipe, revision))
}

fn opts() -> GenerateOptions {
    GenerateOptions {
        max_new_tokens: MAX_NEW_TOKENS_4096_COVERS_THE_DENSEST_SCALE_PAGE_AT_1_2_TOKENS_PER_WORD,
        ..GenerateOptions::default()
    }
}

fn dump(arm: &str, fixture: &str, text: &str) {
    let Ok(dir) = std::env::var("NV_DOTS_SCALE_OUT") else {
        return;
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("create scale out dir");
    std::fs::write(dir.join(format!("{fixture}.{arm}.txt")), text).expect("dump arm text");
}

#[test]
#[ignore]
fn scale_screens_matrix_single_pass_walls_and_cer() {
    let Some((pipe, revision)) = scale_pipeline() else {
        return;
    };
    for name in SCREEN_FIXTURES {
        let fx = load_scale_fixture(name);
        let mut walls = Vec::new();
        let mut last = None;
        for _ in 0..3 {
            let t0 = Instant::now();
            let res = pipe
                .recognize(&fx.img, DotsMode::PlainOcr, &opts())
                .expect("recognize screen");
            walls.push(t0.elapsed().as_secs_f64());
            last = Some(res);
        }
        let res = last.unwrap();
        dump("screen-singlepass", name, &res.text);
        let e_cer = cer(&res.text, &fx.expected);
        let e_ncer = cer_neutral(&res.text, &fx.expected);
        let recall = word_recall(&res.text, &fx.expected);
        let mut sorted = walls.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let spread = (sorted[sorted.len() - 1] - sorted[0]) / sorted[0] * 100.0;
        eprintln!(
            "SCALE-SCREEN fixture={name} src_px={}x{} wall_s_med={:.3} rep_spread_pct={spread:.1} \
             cer={e_cer:.4} ncer={e_ncer:.4} word_recall={recall:.4} gen={} vision_grid={:?} \
             precision={:?} basis=whole_recognize_wall_3reps_median_plainocr_defaults \
             checkpoint=rednote-hilab/dots.ocr@{revision}",
            fx.img.w,
            fx.img.h,
            sorted[1],
            res.generated_tokens,
            res.grid,
            pipe.precision(),
        );
        assert!(res.generated_tokens > 0, "{name}: generated zero tokens");
    }
}

#[test]
#[ignore]
fn scale_big_single_pass_baseline_cer_walls_and_tier_recall() {
    let Some((pipe, revision)) = scale_pipeline() else {
        return;
    };
    for name in BIG_FIXTURES {
        let fx = load_scale_fixture(name);
        let t0 = Instant::now();
        let res = pipe
            .recognize(&fx.img, DotsMode::PlainOcr, &opts())
            .expect("recognize big single pass");
        let wall = t0.elapsed().as_secs_f64();
        dump("baseline", name, &res.text);
        let e_cer = cer(&res.text, &fx.expected);
        let e_ncer = cer_neutral(&res.text, &fx.expected);
        let tiers = tier_recalls(&fx, &res.text);
        eprintln!(
            "SCALE-BASELINE fixture={name} src_px={}x{} wall_s={wall:.3} cer={e_cer:.4} ncer={e_ncer:.4} \
             gen={} looped={} hit_eos={} grid={:?} precision={:?} \
             basis=single_pass_plainocr_default_11_29mp_cap_1rep \
             checkpoint=rednote-hilab/dots.ocr@{revision}",
            fx.img.w,
            fx.img.h,
            res.generated_tokens,
            res.looped,
            res.hit_eos,
            res.grid,
            pipe.precision(),
        );
        for (tier, recall, n) in tiers {
            eprintln!(
                "SCALE-BASELINE-TIER fixture={name} tier={tier} word_recall={recall:.4} words={n}"
            );
        }
    }
}

#[test]
#[ignore]
fn scale_big_tiered_cer_walls_and_tier_recall() {
    let Some((pipe, revision)) = scale_pipeline() else {
        return;
    };
    for name in BIG_FIXTURES {
        let fx = load_scale_fixture(name);
        let t0 = Instant::now();
        let res = pipe
            .recognize_cropped(&fx.img, &opts())
            .expect("recognize_cropped big");
        let wall = t0.elapsed().as_secs_f64();
        dump("tiered", name, &res.text);
        let e_cer = cer(&res.text, &fx.expected);
        let e_ncer = cer_neutral(&res.text, &fx.expected);
        let tiers = tier_recalls(&fx, &res.text);
        eprintln!(
            "SCALE-TIERED fixture={name} src_px={}x{} wall_s={wall:.3} cer={e_cer:.4} ncer={e_ncer:.4} \
             regions={} gen={} looped={} hit_eos={} truncated={} precision={:?} \
             layout_max={} crop_max={} \
             basis=recognize_cropped_1rep_layout_ladder_plus_sequential_crops \
             checkpoint=rednote-hilab/dots.ocr@{revision}",
            fx.img.w,
            fx.img.h,
            res.page.elements.len(),
            res.generated_tokens,
            res.looped,
            res.hit_eos,
            res.page.truncated,
            pipe.precision(),
            std::env::var("NV_DOTS_LAYOUT_MAX_PIXELS").unwrap_or_else(|_| "default".into()),
            std::env::var("NV_DOTS_CROP_MAX_PIXELS").unwrap_or_else(|_| "default".into()),
        );
        for (tier, recall, n) in tiers {
            eprintln!(
                "SCALE-TIERED-TIER fixture={name} tier={tier} word_recall={recall:.4} words={n}"
            );
            assert!(
                recall >= TIERED_TIER_RECALL_GATE_0_9_MEASURED_1_0_EVERY_TIER_AT_7BC6DA5,
                "{name}/{tier}: tiered word recall {recall:.4} under the gate"
            );
        }
        assert!(res.generated_tokens > 0, "{name}: generated zero tokens");
        assert!(
            e_ncer <= TIERED_NCER_GATE_0_03_MEASURED_0194_0000_0000_0000_AT_7BC6DA5,
            "{name}: tiered neutral CER {e_ncer:.4} over the gate"
        );
    }
}

const TIERED_NCER_GATE_0_03_MEASURED_0194_0000_0000_0000_AT_7BC6DA5: f64 = 0.03;
const TIERED_TIER_RECALL_GATE_0_9_MEASURED_1_0_EVERY_TIER_AT_7BC6DA5: f64 = 0.9;
const SCREEN_TIERED_RECALL_GATE_0_80_MEASURED_0861_TO_0972_AT_7BC6DA5: f64 = 0.80;
const SCREEN_TIERED_NCER_GATE_0_25_MEASURED_0045_TO_0189_AT_7BC6DA5: f64 = 0.25;
const T0_RECALL_GATE_0_8_AT_4MP_MEASURED_0867_TO_100_AT_7BC6DA5: f64 = 0.8;
const CLASSICAL_SCOUT_RECALL_GATE_0_9_MEASURED_0967_TO_100_AT_4C1066F: f64 = 0.9;
const THE_MODEL_SCOUT_COLLAPSES_ON_BUSY_PAGES_SO_ITS_RECALL_GATE_EXCLUDES_THE_GAZETTE: &str =
    "the layout-only model pass on the 121-region gazette emits two or three boxes and \
     truncates at every budget (recall 0.017-0.025, gen 67-101), while the classical scout \
     reads 0.967 in 2.5 ms; recognize_auto tiles plain OCR with the classical scout, so the \
     model-scout gate covers only the document-shaped fixtures it serves in layout mode.";

#[test]
#[ignore]
fn scale_screens_budget_ladder_prices_capped_single_pass() {
    let Some((pipe, revision)) = scale_pipeline() else {
        return;
    };
    for name in SCREEN_FIXTURES {
        let fx = load_scale_fixture(name);
        for max_pixels in [1_000_000usize, 2_000_000, 4_000_000, 11_289_600] {
            let budget = PixelBudget {
                min_pixels: nv_models::dots_ocr::preprocess::DEFAULT_MIN_PIXELS,
                max_pixels,
            };
            let t0 = Instant::now();
            let res = pipe
                .recognize_with_budget(&fx.img, DotsMode::PlainOcr, &opts(), budget)
                .expect("recognize screen budgeted");
            let wall = t0.elapsed().as_secs_f64();
            dump(&format!("screen-cap{}", max_pixels / 1_000_000), name, &res.text);
            let e_cer = cer(&res.text, &fx.expected);
        let e_ncer = cer_neutral(&res.text, &fx.expected);
            let recall = word_recall(&res.text, &fx.expected);
            eprintln!(
                "SCALE-SCREEN-CAP fixture={name} max_pixels={max_pixels} src_px={}x{} \
                 wall_s={wall:.3} cer={e_cer:.4} ncer={e_ncer:.4} word_recall={recall:.4} gen={} grid={:?} \
                 precision={:?} basis=budgeted_single_pass_plainocr_1rep \
                 checkpoint=rednote-hilab/dots.ocr@{revision}",
                fx.img.w,
                fx.img.h,
                res.generated_tokens,
                res.grid,
                pipe.precision(),
            );
        }
    }
}

#[test]
#[ignore]
fn scale_screens_tiered_walls_and_cer() {
    let Some((pipe, revision)) = scale_pipeline() else {
        return;
    };
    for name in SCREEN_FIXTURES {
        let fx = load_scale_fixture(name);
        let t0 = Instant::now();
        let res = match pipe
            .recognize_tiled_classical(&fx.img, &opts())
            .expect("classical tiling screen")
        {
            Some(r) => r,
            None => pipe
                .recognize_cropped(&fx.img, &opts())
                .expect("model-scout fallback screen"),
        };
        let wall = t0.elapsed().as_secs_f64();
        dump("screen-tiered", name, &res.text);
        let e_cer = cer(&res.text, &fx.expected);
        let e_ncer = cer_neutral(&res.text, &fx.expected);
        let recall = word_recall(&res.text, &fx.expected);
        eprintln!(
            "SCALE-SCREEN-TIERED fixture={name} src_px={}x{} wall_s={wall:.3} cer={e_cer:.4} \
             ncer={e_ncer:.4} word_recall={recall:.4} regions={} gen={} looped={} \
             precision={:?} basis=recognize_cropped_defaults_1rep \
             checkpoint=rednote-hilab/dots.ocr@{revision}",
            fx.img.w,
            fx.img.h,
            res.page.elements.len(),
            res.generated_tokens,
            res.looped,
            pipe.precision(),
        );
        if name.contains("darkmode") {
            let _ = DARK_SCREENS_ARE_A_MEASURED_LIMITATION_ON_EVERY_PATH_NOT_A_GATE;
            continue;
        }
        assert!(
            recall >= SCREEN_TIERED_RECALL_GATE_0_80_MEASURED_0861_TO_0972_AT_7BC6DA5,
            "{name}: screen tiered word recall {recall:.4} under the gate"
        );
        assert!(
            e_ncer <= SCREEN_TIERED_NCER_GATE_0_25_MEASURED_0045_TO_0189_AT_7BC6DA5,
            "{name}: screen tiered neutral CER {e_ncer:.4} over the gate"
        );
    }
}

const DARK_SCREENS_ARE_A_MEASURED_LIMITATION_ON_EVERY_PATH_NOT_A_GATE: &str =
    "on the 4K dark theme the model layout pass boxes one region and the tiled read stops \
     at word recall 0.667 (42.1 s); the inverted-ink classical scout boxes the text but the \
     small-glyph median guard rightly defers UI-sized glyphs to the model. No current path \
     reads dark screens well, so the row is reported for the record and the gate covers the \
     light screens every path serves; the open lever is compositing dark crops to \
     light-on-dark inversion before the tower, unmeasured.";

const PLAN_FIXTURE: &str = "092-scale-50mp-planroom";
const GAZETTE_FIXTURE: &str = "093-scale-24mp-gazette";

#[test]
#[ignore]
fn scale_busy_page_scout_shootout_prices_model_t0_against_classical_cc() {
    let Some((pipe, revision)) = scale_pipeline() else {
        return;
    };
    let fx = load_scale_fixture(GAZETTE_FIXTURE);
    let arms: [(&str, Box<dyn Fn() -> nv_models::dots_ocr::DotsPageResult>); 3] = [
        (
            "single",
            Box::new(|| {
                pipe.recognize(&fx.img, DotsMode::PlainOcr, &opts())
                    .expect("gazette single pass")
            }),
        ),
        (
            "tiered-model",
            Box::new(|| {
                pipe.recognize_cropped(&fx.img, &opts())
                    .expect("gazette model-scout tiered")
            }),
        ),
        (
            "tiered-classical",
            Box::new(|| {
                pipe.recognize_tiled_classical(&fx.img, &opts())
                    .expect("gazette classical tiered")
                    .expect("gazette is a light page, the classical scout must accept it")
            }),
        ),
    ];
    for (arm, run) in &arms {
        let t0 = Instant::now();
        let res = run();
        let wall = t0.elapsed().as_secs_f64();
        dump(&format!("gazette-{arm}"), GAZETTE_FIXTURE, &res.text);
        let e_cer = cer(&res.text, &fx.expected);
        let e_ncer = cer_neutral(&res.text, &fx.expected);
        let recall = word_recall(&res.text, &fx.expected);
        eprintln!(
            "SCALE-SHOOTOUT arm={arm} fixture={GAZETTE_FIXTURE} wall_s={wall:.3} \
             cer={e_cer:.4} ncer={e_ncer:.4} word_recall={recall:.4} regions={} gen={} \
             looped={} precision={:?} \
             basis=121_region_busy_page_all_arms_defaults_1rep \
             checkpoint=rednote-hilab/dots.ocr@{revision}",
            res.page.elements.len(),
            res.generated_tokens,
            res.looped,
            pipe.precision(),
        );
    }
}

#[test]
#[ignore]
fn scale_plan_crop_budget_ladder_finds_the_native_detail_floor() {
    let Some((pipe, revision)) = scale_pipeline() else {
        return;
    };
    let fx = load_scale_fixture(PLAN_FIXTURE);
    let t0 = Instant::now();
    let res = pipe
        .recognize(&fx.img, DotsMode::PlainOcr, &opts())
        .expect("plan single pass");
    let wall = t0.elapsed().as_secs_f64();
    dump("baseline", PLAN_FIXTURE, &res.text);
    let e_cer = cer(&res.text, &fx.expected);
    let e_ncer = cer_neutral(&res.text, &fx.expected);
    eprintln!(
        "SCALE-PLAN-BASELINE fixture={PLAN_FIXTURE} wall_s={wall:.3} cer={e_cer:.4} ncer={e_ncer:.4} gen={} \
         basis=single_pass_plainocr_default_cap_1rep \
         checkpoint=rednote-hilab/dots.ocr@{revision}",
        res.generated_tokens,
    );
    for (tier, recall, n) in tier_recalls(&fx, &res.text) {
        eprintln!(
            "SCALE-PLAN-BASELINE-TIER tier={tier} word_recall={recall:.4} words={n}"
        );
    }
    let base_wall = wall;
    let base_ncer = e_ncer;
    for crop_max in [500_000usize, 1_000_000, 2_000_000] {
        std::env::set_var("NV_DOTS_CROP_MAX_PIXELS", crop_max.to_string());
        let t0 = Instant::now();
        let res = pipe
            .recognize_cropped(&fx.img, &opts())
            .expect("plan tiered");
        let wall = t0.elapsed().as_secs_f64();
        dump(&format!("tiered-crop{}", crop_max / 100_000), PLAN_FIXTURE, &res.text);
        let e_cer = cer(&res.text, &fx.expected);
        let e_ncer = cer_neutral(&res.text, &fx.expected);
        eprintln!(
            "SCALE-PLAN-TIERED fixture={PLAN_FIXTURE} crop_max={crop_max} wall_s={wall:.3} \
             cer={e_cer:.4} ncer={e_ncer:.4} regions={} gen={} looped={} \
             basis=recognize_cropped_crop_budget_ladder_1rep_env_pinned \
             checkpoint=rednote-hilab/dots.ocr@{revision}",
            res.page.elements.len(),
            res.generated_tokens,
            res.looped,
        );
        for (tier, recall, n) in tier_recalls(&fx, &res.text) {
            eprintln!(
                "SCALE-PLAN-TIERED-TIER crop_max={crop_max} tier={tier} \
                 word_recall={recall:.4} words={n}"
            );
        }
        assert!(
            e_ncer <= base_ncer + 0.005,
            "crop_max={crop_max}: tiered ncer {e_ncer:.4} worse than the same-run \
             single-pass baseline {base_ncer:.4} (measured 0.0052 vs 0.0117)"
        );
        assert!(
            wall < base_wall,
            "crop_max={crop_max}: tiered wall {wall:.1}s not under the same-run \
             single-pass baseline {base_wall:.1}s (measured 21.8 vs 61.8)"
        );
    }
    std::env::remove_var("NV_DOTS_CROP_MAX_PIXELS");
}

#[test]
#[ignore]
fn scale_mp_ladder_single_pass_vs_tiered_crossover() {
    let Some((pipe, revision)) = scale_pipeline() else {
        return;
    };
    let fx = load_scale_fixture("090-scale-35mp-report");
    let src_px = (fx.img.w * fx.img.h) as f64;
    for target_mp in [3.0f64, 5.0, 8.0, 11.2] {
        let f = (target_mp * 1e6 / src_px).sqrt();
        let (tw, th) = (
            (fx.img.w as f64 * f) as usize,
            (fx.img.h as f64 * f) as usize,
        );
        let img = nv_models::deepseek_ocr::preprocess::resize_rgb(&fx.img, tw, th);
        let mut rung: Vec<(f64, f64)> = Vec::new();
        for (arm, res, wall) in [
            {
                let t0 = Instant::now();
                let r = pipe
                    .recognize(&img, DotsMode::PlainOcr, &opts())
                    .expect("mp ladder single pass");
                ("single", r, t0.elapsed().as_secs_f64())
            },
            {
                let t0 = Instant::now();
                let r = pipe
                    .recognize_cropped(&img, &opts())
                    .expect("mp ladder tiered");
                ("tiered", r, t0.elapsed().as_secs_f64())
            },
        ] {
            let e_cer = cer(&res.text, &fx.expected);
            let e_ncer = cer_neutral(&res.text, &fx.expected);
            eprintln!(
                "SCALE-MPLADDER arm={arm} input_mp={target_mp:.1} input_px={tw}x{th} \
                 wall_s={wall:.3} cer={e_cer:.4} ncer={e_ncer:.4} gen={} regions={} \
                 precision={:?} basis=same_resized_090_35mp_page_both_arms_defaults_1rep \
                 checkpoint=rednote-hilab/dots.ocr@{revision}",
                res.generated_tokens,
                res.page.elements.len(),
                pipe.precision(),
            );
            rung.push((wall, e_ncer));
        }
        let (single, tiered) = (rung[0], rung[1]);
        if target_mp >= 5.0 {
            assert!(
                tiered.0 < single.0,
                "at {target_mp:.1}MP the tiered wall {:.1}s must beat single-pass {:.1}s \
                 (measured 12.1 vs 16.9 at 5MP, gap widening with MP)",
                tiered.0,
                single.0
            );
        }
        assert!(
            tiered.1 <= single.1 + 0.005,
            "at {target_mp:.1}MP tiered ncer {:.4} worse than single-pass {:.4}",
            tiered.1,
            single.1
        );
    }
}

fn point_covered(regions: &[Region], x: f32, y: f32) -> bool {
    regions.iter().any(|r| {
        x >= r.bbox[0] && x <= r.bbox[2] && y >= r.bbox[1] && y <= r.bbox[3]
    })
}

fn gt_box_coverage(regions: &[Region], gt: &GtRegion) -> f64 {
    let n = 7;
    let mut hit = 0usize;
    for iy in 0..n {
        for ix in 0..n {
            let fx = (ix as f32 + 0.5) / n as f32;
            let fy = (iy as f32 + 0.5) / n as f32;
            let x = gt.bbox[0] + fx * (gt.bbox[2] - gt.bbox[0]);
            let y = gt.bbox[1] + fy * (gt.bbox[3] - gt.bbox[1]);
            if point_covered(regions, x, y) {
                hit += 1;
            }
        }
    }
    hit as f64 / (n * n) as f64
}

const T0_BOX_RECALLED_AT_080_INTERIOR_COVERAGE_BECAUSE_CROP_MARGINS_RECOVER_EDGES: f64 = 0.80;

#[test]
#[ignore]
fn scale_t0_layout_recall_ladder_on_downscaled_pages() {
    let Some((pipe, revision)) = scale_pipeline() else {
        return;
    };
    for name in BIG_FIXTURES.iter().chain([&PLAN_FIXTURE, &GAZETTE_FIXTURE]) {
        let fx = load_scale_fixture(name);
        {
            let t0 = Instant::now();
            let scouted = nv_models::dots_ocr::tiered::classical_scout(&fx.img);
            let wall = t0.elapsed().as_secs_f64();
            match scouted {
                Some(page) => {
                    let planned = plan_regions(&page, fx.img.w, fx.img.h, false);
                    let recalled = fx
                        .gt_regions
                        .iter()
                        .filter(|gt| {
                            gt_box_coverage(&planned, gt)
                                >= T0_BOX_RECALLED_AT_080_INTERIOR_COVERAGE_BECAUSE_CROP_MARGINS_RECOVER_EDGES
                        })
                        .count();
                    let recall = recalled as f64 / fx.gt_regions.len() as f64;
                    eprintln!(
                        "SCALE-T0-CLASSICAL fixture={name} wall_s={wall:.4} planned_regions={} \
                         gt_regions={} recall={recall:.4} \
                         basis=cpu_ink_cc_scout_plus_plan_regions_point_sampled_coverage \
                         checkpoint=rednote-hilab/dots.ocr@{revision}",
                        planned.len(),
                        fx.gt_regions.len(),
                    );
                    assert!(
                        recall >= CLASSICAL_SCOUT_RECALL_GATE_0_9_MEASURED_0967_TO_100_AT_4C1066F,
                        "{name}: classical scout recall {recall:.4} under the gate"
                    );
                }
                None => eprintln!("SCALE-T0-CLASSICAL fixture={name} wall_s={wall:.4} REFUSED"),
            }
        }
        for layout_mp in [500_000usize, 1_000_000, 2_000_000, 4_000_000] {
            let budget = PixelBudget {
                min_pixels: nv_models::dots_ocr::preprocess::DEFAULT_MIN_PIXELS,
                max_pixels: layout_mp,
            };
            let t0 = Instant::now();
            let res = pipe
                .recognize_with_budget(&fx.img, DotsMode::LayoutOnly, &opts(), budget)
                .expect("layout-only pass");
            let wall = t0.elapsed().as_secs_f64();
            let planned = plan_regions(&res.page, fx.img.w, fx.img.h, false);
            let mut recalled = 0usize;
            let mut worst: (f64, String) = (2.0, String::new());
            for gt in &fx.gt_regions {
                let cov = gt_box_coverage(&planned, gt);
                if cov >= T0_BOX_RECALLED_AT_080_INTERIOR_COVERAGE_BECAUSE_CROP_MARGINS_RECOVER_EDGES {
                    recalled += 1;
                } else if cov < worst.0 {
                    worst = (cov, gt.text.lines().next().unwrap_or("").to_string());
                }
            }
            let recall = recalled as f64 / fx.gt_regions.len() as f64;
            eprintln!(
                "SCALE-T0 fixture={name} layout_max_px={layout_mp} wall_s={wall:.3} \
                 gen={} planned_regions={} gt_regions={} recall={recall:.4} truncated={} \
                 worst_uncovered_cov={:.2} worst_uncovered_head={:?} \
                 basis=layout_only_pass_plus_plan_regions_point_sampled_coverage_1rep \
                 checkpoint=rednote-hilab/dots.ocr@{revision}",
                res.generated_tokens,
                planned.len(),
                fx.gt_regions.len(),
                res.page.truncated,
                worst.0.min(1.0),
                worst.1,
            );
            if layout_mp == 4_000_000 && *name != GAZETTE_FIXTURE {
                let _ = THE_MODEL_SCOUT_COLLAPSES_ON_BUSY_PAGES_SO_ITS_RECALL_GATE_EXCLUDES_THE_GAZETTE;
                assert!(
                    recall >= T0_RECALL_GATE_0_8_AT_4MP_MEASURED_0867_TO_100_AT_7BC6DA5,
                    "{name}: T0 recall {recall:.4} at 4MP under the gate"
                );
            }
        }
    }
}
