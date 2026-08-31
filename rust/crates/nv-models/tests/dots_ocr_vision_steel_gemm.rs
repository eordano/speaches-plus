#![cfg(feature = "wgpu")]

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use nv_kernels::wgpu_backend::WgpuContext;
use nv_models::deepseek_ocr::preprocess::RgbImage;
use nv_models::dots_ocr::preprocess::{prepare, PixelBudget};
use nv_models::dots_ocr::vision::DotsVisionConfig;
use nv_models::dots_ocr::vision_attn_msl::STEEL_GEMM_CONFIGS;
use nv_models::dots_ocr::vision_wgpu::{
    pad_to, DotsVisionTowerWgpu, VisionGemm, VisionGemmRoute, DOTS_VISION_STEEL_GEMM_ENV,
};
use nv_weights::WeightLoader;

const LADDER_REPEATS: usize = 8;

const THE_LADDER_PRICES_STEEL_GEMM_AGAINST_THE_COOP_INCUMBENT_AT_THE_TOWERS_OWN_PROJECTION_SHAPES: &str =
    "The rival's f16-cast tower says our projections leave ~0.4-0.5s against steel-gemm-at-f16, \
     and this arm prices that per shape: the coop incumbent (shipped plan resolution) against \
     every vendored steel config, both timed to a readback at the real row count. A steel config \
     that cannot serve a shape (k not a multiple of its k-block) is skipped, not zeroed.";

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
fn the_steel_gemm_is_priced_per_projection_shape_against_the_coop_incumbent() {
    let _ = THE_LADDER_PRICES_STEEL_GEMM_AGAINST_THE_COOP_INCUMBENT_AT_THE_TOWERS_OWN_PROJECTION_SHAPES;
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    eprintln!("[dv-steel-gemm] adapter: {}", ctx.summary());
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    assert!(route.is_coop(), "the incumbent is the coop route");
    let tile = route.tile();
    let cfg_model = DotsVisionConfig::default();
    let g = VisionGemm::new(ctx, route).expect("vision gemm");
    let m = std::env::var("NV_DOTS_VISION_WGPU_PATCHES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(13600);
    let m_pad = pad_to(m, tile);
    let (dim, inter) = (cfg_model.embed_dim, cfg_model.intermediate_size);
    let merged = dim * 4;
    let shapes: [(&str, usize, usize); 7] = [
        ("qkv", dim * 3, dim),
        ("attn_proj", dim, dim),
        ("mlp_fc1_fc3", inter, dim),
        ("mlp_fc2", dim, inter),
        ("patch_proj", dim, pad_to(784, tile)),
        ("merger_fc0", merged, merged),
        ("merger_fc2", cfg_model.hidden_size, merged),
    ];
    let mut steel_ran = 0usize;
    for (name, n, k_pad) in shapes {
        let flop = 2.0 * m as f64 * n as f64 * k_pad as f64;
        let w = g.f16_buf("dv-sg-w", n * k_pad);
        let x = g.f16_buf("dv-sg-x", g.coop_x_rows(m_pad) * k_pad);
        let y = g.f32_buf("dv-sg-y", m_pad * n);
        g.gemm("projection", &w, 0, &x, 0, &y, 0, m_pad, n, k_pad, n)
            .expect("coop warm");
        g.read_f32(&y, 1).expect("settle");
        let t0 = Instant::now();
        for _ in 0..LADDER_REPEATS {
            g.gemm("projection", &w, 0, &x, 0, &y, 0, m_pad, n, k_pad, n)
                .expect("coop timed");
        }
        g.read_f32(&y, 1).expect("settle");
        let coop = t0.elapsed().as_secs_f64() / LADDER_REPEATS as f64;
        eprintln!(
            "[dv-steel-gemm] {name} m={m} n={n} k={k_pad} coop {:.4} ms {:.2} TFLOP/s",
            coop * 1e3,
            flop / coop / 1e12
        );
        for cfg in STEEL_GEMM_CONFIGS {
            if !g
                .gemm_steel(&w, &x, &y, m, n, k_pad, n, *cfg)
                .expect("steel warm")
            {
                continue;
            }
            g.read_f32(&y, 1).expect("settle");
            let t1 = Instant::now();
            for _ in 0..LADDER_REPEATS {
                g.gemm_steel(&w, &x, &y, m, n, k_pad, n, *cfg)
                    .expect("steel timed");
            }
            g.read_f32(&y, 1).expect("settle");
            let secs = t1.elapsed().as_secs_f64() / LADDER_REPEATS as f64;
            steel_ran += 1;
            eprintln!(
                "[dv-steel-gemm] {name} m={m} n={n} k={k_pad} steel bm{}bn{}bk{}wm{}wn{} {:.4} ms {:.2} TFLOP/s {:.2}x-coop",
                cfg.0, cfg.1, cfg.2, cfg.3, cfg.4,
                secs * 1e3,
                flop / secs / 1e12,
                coop / secs
            );
        }
    }
    assert!(
        steel_ran > 0,
        "no steel config dispatched anywhere, so this ladder priced nothing new"
    );
}

#[test]
#[ignore]
fn the_steel_gemm_is_priced_by_the_forward_wall_and_its_reassociation_is_bounded() {
    let page = std::env::var("NV_DOTS_PAGE")
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let img = RgbImage::decode_file(&page).expect("decode page");
    let prep = prepare(&img, PixelBudget::from_env()).expect("prepare page");
    let cfg = DotsVisionConfig::default();
    let arm = std::env::var("NV_DOTS_STEEL_GEMM_WALL_ARM").unwrap_or_else(|_| "1".into());
    eprintln!(
        "[dv-steel-gemm-wall] page={} grid {}x{} = {} patches, arm={arm}",
        page.display(),
        prep.grid_h,
        prep.grid_w,
        prep.num_patches()
    );
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    match nv_models::dots_ocr::vision_attn_msl::steel_gemm_pipelines(ctx) {
        Ok(p) => eprintln!("[dv-steel-gemm-wall] pipelines ready ({} configs)", p.by_config.len()),
        Err(e) => panic!("steel gemm pipelines failed in this process: {e:#}"),
    }
    let loader = WeightLoader::open_dir(&snapshot_dir().expect("snapshot"), &Device::Cpu)
        .expect("open weights");
    let tower =
        DotsVisionTowerWgpu::from_loader(&loader, "vision_tower.", cfg.clone()).expect("tower load");

    std::env::set_var(DOTS_VISION_STEEL_GEMM_ENV, "0");
    let coop_out = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("coop warmup");
    std::env::set_var(DOTS_VISION_STEEL_GEMM_ENV, arm.as_str());
    let steel_out = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("steel warmup");
    assert!(steel_out.iter().all(|v| v.is_finite()));
    let mut num = 0f64;
    let mut den = 0f64;
    let mut moved = 0usize;
    for (a, b) in coop_out.iter().zip(steel_out.iter()) {
        num += ((a - b) as f64).powi(2);
        den += (*a as f64).powi(2);
        moved += usize::from(a.to_bits() != b.to_bits());
    }
    eprintln!(
        "[dv-steel-gemm-wall] rel-rms vs coop {:.4e}, {moved} of {} merged features moved",
        (num / den.max(1e-30)).sqrt(),
        coop_out.len()
    );
    assert_eq!(
        moved, 0,
        "steel nt and the coop kernel are the same ascending-k chain of 8x8 simdgroup MMAs with \
         f32 accumulators, so any differing bit means the steel arm is not the same function and \
         cannot ship as a free win"
    );

    let reps: usize = 3;
    let mut w_c = Vec::new();
    let mut w_s = Vec::new();
    for rep in 0..reps {
        for st in [false, true] {
            std::env::set_var(
                DOTS_VISION_STEEL_GEMM_ENV,
                if st { arm.as_str() } else { "0" },
            );
            let t = Instant::now();
            tower
                .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
                .expect("timed forward");
            let wall = t.elapsed().as_secs_f64();
            if st {
                w_s.push(wall);
            } else {
                w_c.push(wall);
            }
            eprintln!(
                "[dv-steel-gemm-wall] rep {rep} {} {wall:.3}s",
                if st { "steel-gemm" } else { "coop" }
            );
        }
    }
    std::env::remove_var(DOTS_VISION_STEEL_GEMM_ENV);
    let mean = |w: &[f64]| w.iter().sum::<f64>() / w.len() as f64;
    let spread = |w: &[f64], mm: f64| w.iter().fold(0f64, |a, v| a.max((v - mm).abs()));
    let (mc, ms) = (mean(&w_c), mean(&w_s));
    eprintln!(
        "[dv-steel-gemm-wall] coop mean {mc:.3}s +-{:.3}s, steel-gemm({arm}) mean {ms:.3}s +-{:.3}s, delta {:.3}s = {:.1}%",
        spread(&w_c, mc),
        spread(&w_s, ms),
        mc - ms,
        (mc - ms) / mc * 100.0
    );
}
