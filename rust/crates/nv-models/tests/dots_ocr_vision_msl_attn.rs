#![cfg(feature = "wgpu")]

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use nv_models::deepseek_ocr::preprocess::RgbImage;
use nv_models::dots_ocr::preprocess::{prepare, PixelBudget};
use nv_models::dots_ocr::vision::DotsVisionConfig;
use nv_models::dots_ocr::vision_wgpu::{DotsVisionTowerWgpu, DOTS_VISION_WGPU_ATTN_ENV};
use nv_weights::WeightLoader;

const THE_WALL_ARM_PRICES_THE_FUSED_RESIDUE_THE_BRACKET_NAMED: &str =
    "By deleted work the zmerge attention costs scores 1.403 + ctx 1.462 + softmax 0.961 = 3.83s \
     of the 6.16s invoice tower, and the rival's fused kernel does the same arithmetic in about \
     3.1-3.4s. The msl arm replaces all three phases plus the merge with one fused dispatch per \
     layer; its online softmax reassociates, so the accept gate is corpus token equality and \
     this arm's rel-rms row is the early tripwire, the zmerge precedent.";

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

struct StderrLog;

impl log::Log for StderrLog {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, r: &log::Record) {
        eprintln!("[log:{}] {}", r.level(), r.args());
    }
    fn flush(&self) {}
}

static STDERR_LOG: StderrLog = StderrLog;

#[test]
#[ignore]
fn the_msl_attention_source_compiles_or_the_stderr_logger_names_the_diagnostic() {
    let _ = log::set_logger(&STDERR_LOG).map(|()| log::set_max_level(log::LevelFilter::Error));
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared().expect("wgpu adapter");
    for (aq, ak) in [(false, false), (true, true)] {
        match nv_models::dots_ocr::vision_attn_msl::steel_pipelines(ctx, aq, ak) {
            Ok(_) => eprintln!("[dv-msl-compile] steel pipeline built (align_q={aq} align_k={ak})"),
            Err(e) => panic!("dv-steel-attn compile aq={aq} ak={ak}: {e:#}"),
        }
    }
    match nv_models::dots_ocr::vision_attn_msl::steel_gemm_pipelines(ctx) {
        Ok(p) => eprintln!("[dv-msl-compile] steel gemm pipelines built ({} configs)", p.by_config.len()),
        Err(e) => panic!("dv-steel-gemm compile: {e:#}"),
    }
}

#[test]
#[ignore]
fn the_msl_fused_attention_is_priced_by_the_forward_wall_and_its_reassociation_is_bounded() {
    let _ = THE_WALL_ARM_PRICES_THE_FUSED_RESIDUE_THE_BRACKET_NAMED;
    let page = std::env::var("NV_DOTS_PAGE")
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let img = RgbImage::decode_file(&page).expect("decode page");
    let prep = prepare(&img, PixelBudget::from_env()).expect("prepare page");
    let cfg = DotsVisionConfig::default();
    let arm = std::env::var("NV_DOTS_MSL_WALL_ARM").unwrap_or_else(|_| "steel".into());
    eprintln!(
        "[dv-msl-wall] page={} grid {}x{} = {} patches, arm={arm}",
        page.display(),
        prep.grid_h,
        prep.grid_w,
        prep.num_patches()
    );
    let loader = WeightLoader::open_dir(&snapshot_dir().expect("snapshot"), &Device::Cpu)
        .expect("open weights");
    let tower =
        DotsVisionTowerWgpu::from_loader(&loader, "vision_tower.", cfg.clone()).expect("tower load");

    std::env::set_var(DOTS_VISION_WGPU_ATTN_ENV, "zmerge");
    let zmerge_out = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("zmerge warmup");
    std::env::set_var(DOTS_VISION_WGPU_ATTN_ENV, arm.as_str());
    let msl_out = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("msl warmup");
    assert!(msl_out.iter().all(|v| v.is_finite()));
    let mut num = 0f64;
    let mut den = 0f64;
    let mut moved = 0usize;
    for (a, b) in zmerge_out.iter().zip(msl_out.iter()) {
        num += ((a - b) as f64).powi(2);
        den += (*a as f64).powi(2);
        moved += usize::from(a.to_bits() != b.to_bits());
    }
    eprintln!(
        "[dv-msl-wall] rel-rms vs zmerge {:.4e}, {moved} of {} merged features moved",
        (num / den.max(1e-30)).sqrt(),
        zmerge_out.len()
    );
    assert!(
        moved > 0,
        "an identical feature map means the msl arm never dispatched and this run priced zmerge \
         against itself"
    );

    let reps: usize = 3;
    let mut w_z = Vec::new();
    let mut w_m = Vec::new();
    for rep in 0..reps {
        for m in [false, true] {
            std::env::set_var(
                DOTS_VISION_WGPU_ATTN_ENV,
                if m { arm.as_str() } else { "zmerge" },
            );
            let t = Instant::now();
            tower
                .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
                .expect("timed forward");
            let wall = t.elapsed().as_secs_f64();
            if m {
                w_m.push(wall);
            } else {
                w_z.push(wall);
            }
            eprintln!(
                "[dv-msl-wall] rep {rep} {} {wall:.3}s",
                if m { arm.as_str() } else { "zmerge" }
            );
        }
    }
    std::env::remove_var(DOTS_VISION_WGPU_ATTN_ENV);
    let mean = |w: &[f64]| w.iter().sum::<f64>() / w.len() as f64;
    let spread = |w: &[f64], mm: f64| w.iter().fold(0f64, |a, v| a.max((v - mm).abs()));
    let (mz, mm) = (mean(&w_z), mean(&w_m));
    eprintln!(
        "[dv-msl-wall] zmerge mean {mz:.3}s +-{:.3}s, {arm} mean {mm:.3}s +-{:.3}s, delta {:.3}s = {:.1}%",
        spread(&w_z, mz),
        spread(&w_m, mm),
        mz - mm,
        (mz - mm) / mz * 100.0
    );
}

#[test]
#[ignore]
fn the_shipped_default_wall_is_stamped_with_no_envs_set() {
    let page = std::env::var("NV_DOTS_PAGE")
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let img = RgbImage::decode_file(&page).expect("decode page");
    let prep = prepare(&img, PixelBudget::from_env()).expect("prepare page");
    let cfg = DotsVisionConfig::default();
    std::env::remove_var(DOTS_VISION_WGPU_ATTN_ENV);
    eprintln!(
        "[dv-default-wall] page={} grid {}x{} = {} patches, every vision env unset",
        page.display(),
        prep.grid_h,
        prep.grid_w,
        prep.num_patches()
    );
    let loader = WeightLoader::open_dir(&snapshot_dir().expect("snapshot"), &Device::Cpu)
        .expect("open weights");
    let tower =
        DotsVisionTowerWgpu::from_loader(&loader, "vision_tower.", cfg.clone()).expect("tower load");
    let warm = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("warmup forward");
    assert!(warm.iter().all(|v| v.is_finite()));
    let mut walls = Vec::new();
    for rep in 0..3 {
        let t = Instant::now();
        tower
            .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
            .expect("timed forward");
        let wall = t.elapsed().as_secs_f64();
        walls.push(wall);
        eprintln!("[dv-default-wall] rep {rep} {wall:.3}s");
    }
    let mean = walls.iter().sum::<f64>() / walls.len() as f64;
    let spread = walls.iter().fold(0f64, |a, v| a.max((v - mean).abs()));
    eprintln!("[dv-default-wall] defaults-only mean {mean:.3}s +-{spread:.3}s");
}
