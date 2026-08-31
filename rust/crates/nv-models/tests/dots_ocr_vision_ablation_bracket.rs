#![cfg(feature = "wgpu")]

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use nv_models::deepseek_ocr::preprocess::RgbImage;
use nv_models::dots_ocr::preprocess::{prepare, PixelBudget};
use nv_models::dots_ocr::vision::DotsVisionConfig;
use nv_models::dots_ocr::vision_wgpu::{
    DotsVisionTowerWgpu, DOTS_VISION_ABLATE_ENV, DOTS_VISION_STEEL_GEMM_ENV,
};
use nv_weights::WeightLoader;

const REPS_ENV: &str = "NV_DOTS_VISION_ABLATE_REPS";

const BRACKET_ROWS_PIN_STEEL_GEMM_0_BECAUSE_UNSET_NOW_MEANS_THE_STEEL_DEFAULT: &str =
    "Steel gemm is the shipped projection default where the passthrough probe passes, and the \
     bracket's recorded shares price the coop projection pipeline, so it says \
     NV_DOTS_VISION_STEEL_GEMM=0 explicitly to keep every row's meaning.";

const ARMS_INTERLEAVE_ACROSS_REPS_SO_SLOW_DRIFT_LANDS_IN_EVERY_ARM_EQUALLY: &[&str] = &[
    "control",
    "scores",
    "softmax",
    "ctx",
    "proj",
    "patch_embed",
    "rmsnorm",
    "rope_split",
    "attn_merge",
    "silu",
    "residual_add",
    "trunk_readback",
    "merger",
];

const MERGER_IS_A_SUB_ARM_OF_TRUNK_READBACK_SO_THE_NAMED_SUM_SKIPS_IT: &str =
    "trunk_readback deletes the post-norm, the trunk readback AND the merger behind them, so its \
     delta already contains the merger arm's; summing both would double-count. The merger row is \
     reported to split readback from merger inside that delta, and the leftover is control minus \
     every named arm except merger.";

const THE_BRACKET_READS_THE_WALL_AT_THE_MERGED_FEATURES_READBACK_BECAUSE_A_TIME_THAT_DOES_NOT_END_IN_A_READBACK_MEASURES_SUBMISSION: &str =
    "forward_patches returns only after the merger's read_f32, so its wall is completion, not \
     submission. Each ablation arm deletes dispatches and the share it reports is the wall delta \
     against the interleaved control -- the drained phase table over-attributes (measured 1.36x \
     drain distortion) and per-pass timestamps have inverted a phase map once already, so neither \
     is allowed to aim kernel work.";

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
fn the_deleted_work_bracket_names_which_phase_carries_the_vision_wall() {
    let _ = THE_BRACKET_READS_THE_WALL_AT_THE_MERGED_FEATURES_READBACK_BECAUSE_A_TIME_THAT_DOES_NOT_END_IN_A_READBACK_MEASURES_SUBMISSION;
    let page = std::env::var("NV_DOTS_PAGE")
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let img = RgbImage::decode_file(&page).expect("decode page");
    let prep = prepare(&img, PixelBudget::from_env()).expect("prepare page");
    let cfg = DotsVisionConfig::default();
    let reps: usize = std::env::var(REPS_ENV)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(3);
    eprintln!(
        "[dv-ablate] page={} grid {}x{} = {} patches, {} vision tokens, {reps} reps/arm",
        page.display(),
        prep.grid_h,
        prep.grid_w,
        prep.num_patches(),
        prep.num_vision_tokens()
    );

    let loader = WeightLoader::open_dir(&snapshot_dir().expect("snapshot"), &Device::Cpu)
        .expect("open weights");
    let tower =
        DotsVisionTowerWgpu::from_loader(&loader, "vision_tower.", cfg.clone()).expect("tower load");

    let _ = BRACKET_ROWS_PIN_STEEL_GEMM_0_BECAUSE_UNSET_NOW_MEANS_THE_STEEL_DEFAULT;
    std::env::set_var("NV_DOTS_VISION_WGPU_ATTN", "zmerge");
    std::env::set_var(DOTS_VISION_STEEL_GEMM_ENV, "0");
    std::env::remove_var(DOTS_VISION_ABLATE_ENV);
    let warm = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("warmup forward");
    assert_eq!(warm.len(), prep.num_vision_tokens() * cfg.hidden_size);
    assert!(warm.iter().all(|v| v.is_finite()));

    let arms = ARMS_INTERLEAVE_ACROSS_REPS_SO_SLOW_DRIFT_LANDS_IN_EVERY_ARM_EQUALLY;
    let mut walls: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for rep in 0..reps {
        for (i, arm) in arms.iter().enumerate() {
            if *arm == "control" {
                std::env::remove_var(DOTS_VISION_ABLATE_ENV);
            } else {
                std::env::set_var(DOTS_VISION_ABLATE_ENV, arm);
            }
            let t = Instant::now();
            let out = tower
                .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
                .expect("ablated forward");
            let wall = t.elapsed().as_secs_f64();
            assert_eq!(out.len(), prep.num_vision_tokens() * cfg.hidden_size);
            assert!(
                out.iter().all(|v| v.is_finite()),
                "{arm}: an ablated forward produced a non-finite feature, so its downstream cost \
                 is not value-independent and its wall delta prices nothing"
            );
            walls[i].push(wall);
            eprintln!("[dv-ablate] rep {rep} {arm} {wall:.3}s");
        }
    }
    std::env::remove_var(DOTS_VISION_ABLATE_ENV);

    let stats: Vec<(f64, f64)> = walls
        .iter()
        .map(|w| {
            let mean = w.iter().sum::<f64>() / w.len() as f64;
            let spread = w.iter().fold(0f64, |a, v| a.max((v - mean).abs()));
            (mean, spread)
        })
        .collect();
    let _ = MERGER_IS_A_SUB_ARM_OF_TRUNK_READBACK_SO_THE_NAMED_SUM_SKIPS_IT;
    let control = stats[0].0;
    let mut named = 0.0;
    for (i, arm) in arms.iter().enumerate() {
        let (mean, spread) = stats[i];
        let delta = control - mean;
        if i > 0 && *arm != "merger" {
            named += delta;
        }
        eprintln!(
            "[dv-ablate] {arm}: mean {mean:.3}s spread +-{spread:.3}s{}",
            if i == 0 {
                String::new()
            } else {
                format!(" delta {delta:.3}s = {:.1}% of control", delta / control * 100.0)
            }
        );
    }
    let attn_gemms = (control - stats[1].0) + (control - stats[3].0);
    eprintln!(
        "[dv-ablate] attention GEMMs (scores+ctx) {attn_gemms:.3}s = {:.1}%, residual {:.3}s = {:.1}%",
        attn_gemms / control * 100.0,
        control - named,
        (control - named) / control * 100.0
    );
    assert!(control > 0.0);
    for (i, (mean, _)) in stats.iter().enumerate() {
        assert!(
            i == 0 || *mean <= control * 1.05,
            "{}: deleting work made the wall slower than control by more than the noise floor, so \
             this arm did not delete what it claims",
            arms[i]
        );
    }
}
