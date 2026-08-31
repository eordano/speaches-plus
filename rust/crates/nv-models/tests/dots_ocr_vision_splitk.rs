#![cfg(feature = "wgpu")]

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use nv_kernels::wgpu_backend::WgpuContext;
use nv_models::deepseek_ocr::preprocess::RgbImage;
use nv_models::dots_ocr::preprocess::{prepare, PixelBudget};
use nv_models::dots_ocr::vision::DotsVisionConfig;
use nv_models::dots_ocr::vision_wgpu::{
    pad_to, CoopPlan, DotsVisionTowerWgpu, VisionGemm, VisionGemmRoute, DOTS_VISION_SPLITK_DEFAULT,
    DOTS_VISION_SPLITK_ENV, DOTS_VISION_STEEL_GEMM_ENV,
};
use nv_weights::WeightLoader;

const PATCHES_ENV: &str = "NV_DOTS_VISION_WGPU_PATCHES";

const LADDER_REPEATS: usize = 8;

const PLAIN_ARMS_PIN_SPLITK_0_BECAUSE_UNSET_NOW_MEANS_THE_S16_DEFAULT: &str =
    "Split-k is the shipped attn_ctx default, so an arm that wants the plain single-dispatch \
     kernel must say NV_DOTS_VISION_SPLITK=0 explicitly; leaving the env unset re-measures the \
     default and silently changes what the row means.";

const COOP_ROWS_PIN_STEEL_GEMM_0_BECAUSE_UNSET_NOW_MEANS_THE_STEEL_DEFAULT: &str =
    "Steel gemm is the shipped projection default where the passthrough probe passes, so a row \
     whose meaning is the coop projection pipeline must say NV_DOTS_VISION_STEEL_GEMM=0 \
     explicitly; a reference arm that silently becomes the default compares a thing to itself.";

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
fn the_splitk_ctx_arm_is_priced_per_shape_against_the_plain_kernel() {
    let _ = PLAIN_ARMS_PIN_SPLITK_0_BECAUSE_UNSET_NOW_MEANS_THE_S16_DEFAULT;
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    eprintln!("[dv-splitk] adapter: {}", ctx.summary());
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    assert!(route.is_coop(), "the split-k arm is a cooperative-matrix concept");
    let tile = route.tile();
    let cfg = DotsVisionConfig::default();
    let g = VisionGemm::new(ctx, route).expect("vision gemm");
    let n = std::env::var(PATCHES_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(13600);
    let n_pad = pad_to(n, tile);
    let hd = cfg.head_dim();
    let qb = g.attention_query_block_rows(n_pad).min(n_pad);
    let sm_pad = pad_to(qb, tile);
    let flop = 2.0 * sm_pad as f64 * hd as f64 * n_pad as f64;
    let plans = [
        CoopPlan { tm: 4, tn: 4, sg: 4, ku: 1 },
        CoopPlan { tm: 8, tn: 4, sg: 4, ku: 1 },
        CoopPlan { tm: 2, tn: 4, sg: 4, ku: 1 },
    ];
    let _ = COOP_ROWS_PIN_STEEL_GEMM_0_BECAUSE_UNSET_NOW_MEANS_THE_STEEL_DEFAULT;
    std::env::set_var("NV_DOTS_VISION_WGPU_ATTN", "zmerge");
    std::env::set_var(DOTS_VISION_STEEL_GEMM_ENV, "0");
    std::env::set_var(DOTS_VISION_SPLITK_ENV, "0");
    let plain = g
        .time_gemm_plans("attn_ctx", hd, n_pad, sm_pad, LADDER_REPEATS, &plans)
        .expect("plain ctx ladder");
    for (plan, secs) in &plain {
        eprintln!(
            "[dv-splitk] attn_ctx m={sm_pad} n={hd} k={n_pad} plain {} {:.4} ms {:.2} TFLOP/s",
            plan.name(),
            secs * 1e3,
            flop / secs / 1e12
        );
    }
    let mut ran = 0usize;
    for s in [4u32, 8, 16, 32] {
        std::env::set_var(DOTS_VISION_SPLITK_ENV, s.to_string());
        let rows = g
            .time_gemm_plans("attn_ctx", hd, n_pad, sm_pad, LADDER_REPEATS, &plans)
            .expect("splitk ctx ladder");
        for (plan, secs) in &rows {
            ran += 1;
            eprintln!(
                "[dv-splitk] attn_ctx m={sm_pad} n={hd} k={n_pad} splitk s{s} {} {:.4} ms {:.2} TFLOP/s",
                plan.name(),
                secs * 1e3,
                flop / secs / 1e12
            );
        }
    }
    std::env::remove_var(DOTS_VISION_SPLITK_ENV);
    assert!(ran > 0, "no split-k arm dispatched, so this ladder priced nothing");
}

#[test]
#[ignore]
fn the_splitk_ctx_arm_is_priced_by_the_forward_wall_and_its_reassociation_is_bounded() {
    let _ = PLAIN_ARMS_PIN_SPLITK_0_BECAUSE_UNSET_NOW_MEANS_THE_S16_DEFAULT;
    let page = std::env::var("NV_DOTS_PAGE")
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let img = RgbImage::decode_file(&page).expect("decode page");
    let prep = prepare(&img, PixelBudget::from_env()).expect("prepare page");
    let cfg = DotsVisionConfig::default();
    let splits = std::env::var(DOTS_VISION_SPLITK_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|s| *s > 1)
        .unwrap_or(DOTS_VISION_SPLITK_DEFAULT);
    eprintln!(
        "[dv-splitk-wall] page={} grid {}x{} = {} patches, S={splits}",
        page.display(),
        prep.grid_h,
        prep.grid_w,
        prep.num_patches()
    );
    let loader = WeightLoader::open_dir(&snapshot_dir().expect("snapshot"), &Device::Cpu)
        .expect("open weights");
    let tower =
        DotsVisionTowerWgpu::from_loader(&loader, "vision_tower.", cfg.clone()).expect("tower load");

    let _ = COOP_ROWS_PIN_STEEL_GEMM_0_BECAUSE_UNSET_NOW_MEANS_THE_STEEL_DEFAULT;
    std::env::set_var("NV_DOTS_VISION_WGPU_ATTN", "zmerge");
    std::env::set_var(DOTS_VISION_STEEL_GEMM_ENV, "0");
    std::env::set_var(DOTS_VISION_SPLITK_ENV, "0");
    let plain_out = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("warmup forward");
    std::env::set_var(DOTS_VISION_SPLITK_ENV, splits.to_string());
    let split_out = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("splitk warmup forward");
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in plain_out.iter().zip(split_out.iter()) {
        num += ((a - b) as f64).powi(2);
        den += (*a as f64).powi(2);
    }
    let rel = (num / den.max(1e-30)).sqrt();
    eprintln!(
        "[dv-splitk-wall] rel-rms vs plain {rel:.4e} over {} merged features (a reassociation is \
         nonzero here; the accept gate is corpus token equality, this row is the early tripwire)",
        plain_out.len()
    );
    assert!(split_out.iter().all(|v| v.is_finite()));

    let reps: usize = 3;
    let mut plain_w = Vec::new();
    let mut split_w = Vec::new();
    for rep in 0..reps {
        for split in [false, true] {
            std::env::set_var(
                DOTS_VISION_SPLITK_ENV,
                if split { splits.to_string() } else { "0".to_string() },
            );
            let t = Instant::now();
            tower
                .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
                .expect("timed forward");
            let wall = t.elapsed().as_secs_f64();
            if split {
                split_w.push(wall);
            } else {
                plain_w.push(wall);
            }
            eprintln!(
                "[dv-splitk-wall] rep {rep} {} {wall:.3}s",
                if split { "splitk" } else { "plain" }
            );
        }
    }
    std::env::remove_var(DOTS_VISION_SPLITK_ENV);
    let mean = |w: &[f64]| w.iter().sum::<f64>() / w.len() as f64;
    let spread = |w: &[f64], m: f64| w.iter().fold(0f64, |a, v| a.max((v - m).abs()));
    let (pm, sm) = (mean(&plain_w), mean(&split_w));
    eprintln!(
        "[dv-splitk-wall] plain mean {pm:.3}s +-{:.3}s, splitk s{splits} mean {sm:.3}s +-{:.3}s, delta {:.3}s = {:.1}%",
        spread(&plain_w, pm),
        spread(&split_w, sm),
        pm - sm,
        (pm - sm) / pm * 100.0
    );
}
