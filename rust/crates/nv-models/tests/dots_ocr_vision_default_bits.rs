#![cfg(feature = "wgpu")]

use std::path::PathBuf;

use candle_core::Device;
use nv_models::deepseek_ocr::preprocess::RgbImage;
use nv_models::dots_ocr::preprocess::{prepare, PixelBudget};
use nv_models::dots_ocr::vision::DotsVisionConfig;
use nv_models::dots_ocr::vision_wgpu::DotsVisionTowerWgpu;
use nv_weights::WeightLoader;

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
fn the_default_env_forward_prints_a_bit_fingerprint_so_two_builds_can_be_compared() {
    let page = std::env::var("NV_DOTS_PAGE")
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let want: usize = std::env::var("NV_DOTS_VISION_WGPU_PATCHES")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(1024);
    let img = RgbImage::decode_file(&page).expect("decode page");
    let prep = prepare(
        &img,
        PixelBudget {
            min_pixels: 3136,
            max_pixels: want * 196,
        },
    )
    .expect("prepare page");
    let loader = WeightLoader::open_dir(&snapshot_dir().expect("snapshot"), &Device::Cpu)
        .expect("open weights");
    let tower = DotsVisionTowerWgpu::from_loader(
        &loader,
        "vision_tower.",
        DotsVisionConfig::default(),
    )
    .expect("tower load");
    let out = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("forward");
    let mut h = 0xcbf29ce484222325u64;
    for v in &out {
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    eprintln!(
        "[dv-bits] n={} patches, {} features, fnv={h:016x}, first={:?}",
        prep.num_patches(),
        out.len(),
        &out[..4.min(out.len())]
    );
}
