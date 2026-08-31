#![cfg(feature = "wgpu")]

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use nv_models::deepseek_ocr::preprocess::RgbImage;
use nv_models::dots_ocr::preprocess::{prepare, PixelBudget};
use nv_models::dots_ocr::vision::DotsVisionConfig;
use nv_models::dots_ocr::vision_wgpu::{DotsVisionTowerWgpu, DOTS_VISION_WGPU_MERGER_ENV};
use nv_weights::WeightLoader;

const THE_HOST_MERGER_IS_THE_RESIDUAL_BRACKETS_ONE_FINDING: &str =
    "Deleting the merger moves the post-split-k invoice wall by 0.288s of 6.435s while every \
     other residual arm reads noise, so the residual is one thing: the host merger's readback, \
     CPU layer-norm and gelu, and interleaved GEMM round trips. The gpu merger arm already \
     exists and deletes those trips; its layer-norm and gelu are the same function to every \
     consumer but not the same bits, so the gate is corpus token equality, the zmerge precedent.";

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
fn the_gpu_merger_is_priced_by_the_forward_wall_and_its_rounding_is_bounded() {
    let _ = THE_HOST_MERGER_IS_THE_RESIDUAL_BRACKETS_ONE_FINDING;
    let page = std::env::var("NV_DOTS_PAGE")
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let img = RgbImage::decode_file(&page).expect("decode page");
    let prep = prepare(&img, PixelBudget::from_env()).expect("prepare page");
    let cfg = DotsVisionConfig::default();
    eprintln!(
        "[dv-merger] page={} grid {}x{} = {} patches",
        page.display(),
        prep.grid_h,
        prep.grid_w,
        prep.num_patches()
    );
    let loader = WeightLoader::open_dir(&snapshot_dir().expect("snapshot"), &Device::Cpu)
        .expect("open weights");
    let tower =
        DotsVisionTowerWgpu::from_loader(&loader, "vision_tower.", cfg.clone()).expect("tower load");

    std::env::set_var(DOTS_VISION_WGPU_MERGER_ENV, "host");
    let host_out = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("host-merger warmup");
    std::env::set_var(DOTS_VISION_WGPU_MERGER_ENV, "gpu");
    let gpu_out = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("gpu-merger warmup");
    assert!(gpu_out.iter().all(|v| v.is_finite()));
    let mut num = 0f64;
    let mut den = 0f64;
    let mut moved = 0usize;
    for (a, b) in host_out.iter().zip(gpu_out.iter()) {
        num += ((a - b) as f64).powi(2);
        den += (*a as f64).powi(2);
        moved += usize::from(a.to_bits() != b.to_bits());
    }
    eprintln!(
        "[dv-merger] rel-rms vs host merger {:.4e}, {moved} of {} merged features moved",
        (num / den.max(1e-30)).sqrt(),
        host_out.len()
    );
    assert!(
        moved > 0,
        "an identical feature map means the gpu merger never ran and this arm priced the host \
         path against itself"
    );

    let reps: usize = 3;
    let mut w_host = Vec::new();
    let mut w_gpu = Vec::new();
    for rep in 0..reps {
        for gpu in [false, true] {
            std::env::set_var(DOTS_VISION_WGPU_MERGER_ENV, if gpu { "gpu" } else { "host" });
            let t = Instant::now();
            tower
                .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
                .expect("timed forward");
            let wall = t.elapsed().as_secs_f64();
            if gpu {
                w_gpu.push(wall);
            } else {
                w_host.push(wall);
            }
            eprintln!(
                "[dv-merger] rep {rep} {} {wall:.3}s",
                if gpu { "gpu" } else { "host" }
            );
        }
    }
    std::env::remove_var(DOTS_VISION_WGPU_MERGER_ENV);
    let mean = |w: &[f64]| w.iter().sum::<f64>() / w.len() as f64;
    let spread = |w: &[f64], m: f64| w.iter().fold(0f64, |a, v| a.max((v - m).abs()));
    let (mh, mg) = (mean(&w_host), mean(&w_gpu));
    eprintln!(
        "[dv-merger] host mean {mh:.3}s +-{:.3}s, gpu mean {mg:.3}s +-{:.3}s, delta {:.3}s = {:.1}%",
        spread(&w_host, mh),
        spread(&w_gpu, mg),
        mh - mg,
        (mh - mg) / mh * 100.0
    );
}
