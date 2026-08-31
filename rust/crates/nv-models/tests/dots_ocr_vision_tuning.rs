#![cfg(feature = "wgpu")]

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use nv_kernels::wgpu_backend::WgpuContext;
use nv_models::deepseek_ocr::preprocess::RgbImage;
use nv_models::dots_ocr::preprocess::{prepare, PixelBudget};
use nv_models::dots_ocr::vision::DotsVisionConfig;
use nv_models::dots_ocr::vision_wgpu::{
    mlx_w4_quantize_rows, pad_to, DotsVisionTowerWgpu, VisionGemmRoute, VisionWeightFmt,
    DOTS_VISION_WGPU_ATTN_ENV, DOTS_VISION_WGPU_FITTED_PLANS_ENV, DOTS_VISION_WGPU_FLASH_PLAN_ENV,
    DOTS_VISION_WGPU_MERGER_ENV, DOTS_VISION_WGPU_WQ_ENV,
};
use nv_weights::WeightLoader;

const PAGE_ENV: &str = "NV_DOTS_PAGE";

const PATCHES_ENV: &str = "NV_DOTS_VISION_WGPU_PATCHES";

const PIXELS_PER_PATCH: usize = 196;

const SWEEP_ROUNDS_ARE_TWO_BECAUSE_A_SIBLING_PROCESS_SHARES_THIS_GPU_AND_ONE_SAMPLE_PER_ARM_CANNOT_NAME_ITS_OWN_NOISE: usize =
    2;

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

fn max_abs_rel(got: &[f32], want: &[f32]) -> (f64, f64) {
    let mut mab = 0f64;
    let mut mrel = 0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        let d = (*g as f64 - *w as f64).abs();
        mab = mab.max(d);
        mrel = mrel.max(d / (w.abs() as f64).max(1e-3));
    }
    (mab, mrel)
}

#[test]
fn mlx_w4_quantize_roundtrips_through_the_reference_dequant_within_half_a_bf16_step() {
    let (n, k, gs) = (24usize, 256usize, 64usize);
    let mut v = 0.123f32;
    let w: Vec<f32> = (0..n * k)
        .map(|i| {
            v = (v * 1103.515f32 + 0.619).fract();
            (v - 0.5) * (0.01 + (i % 7) as f32 * 0.02)
        })
        .collect();
    let q = mlx_w4_quantize_rows(&w, n, k, gs);
    let scales: Vec<f32> = q
        .scales_bf16
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let biases: Vec<f32> = q
        .biases_bf16
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let deq =
        nv_quant::mlx_affine::dequantize_mlx_affine_linear(&q.words, &scales, &biases, n, k, gs);
    for (i, (orig, back)) in w.iter().zip(deq.iter()).enumerate() {
        let g = (i % k) / gs;
        let s = scales[(i / k) * (k / gs) + g].abs();
        let bar = s * 0.5 + s / 128.0 + 1e-7;
        assert!(
            (orig - back).abs() <= bar as f32,
            "element {i}: {orig} reconstructs as {back}, past half a quantization step {bar}"
        );
    }
}

#[test]
#[ignore]
fn the_vision_tower_arm_sweep_prices_merger_fitted_flash_and_w4_on_this_adapter() {
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    eprintln!("[dv-sweep] adapter: {}", ctx.summary());
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    let cfg = DotsVisionConfig::default();
    let page = std::env::var(PAGE_ENV)
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let want = std::env::var(PATCHES_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(7396);
    let img = RgbImage::decode_file(&page).expect("decode page");
    let prep = prepare(
        &img,
        PixelBudget {
            min_pixels: 3136,
            max_pixels: want * PIXELS_PER_PATCH,
        },
    )
    .expect("prepare page");
    let n = prep.num_patches();
    eprintln!(
        "[dv-sweep] route={} grid {}x{} = {n} patches (n_pad={})",
        route.name(),
        prep.grid_h,
        prep.grid_w,
        pad_to(n, route.tile())
    );
    let dir = snapshot_dir().expect("dots.ocr snapshot present in the hub cache");
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).expect("open dots.ocr weights");

    for e in [
        DOTS_VISION_WGPU_ATTN_ENV,
        DOTS_VISION_WGPU_FLASH_PLAN_ENV,
        DOTS_VISION_WGPU_MERGER_ENV,
        DOTS_VISION_WGPU_FITTED_PLANS_ENV,
        DOTS_VISION_WGPU_WQ_ENV,
    ] {
        std::env::remove_var(e);
    }
    let t0 = Instant::now();
    let tower = DotsVisionTowerWgpu::from_loader_route(&loader, "vision_tower.", cfg.clone(), route)
        .expect("f16 tower load");
    eprintln!("[dv-sweep] f16 tower staged in {:.1}s", t0.elapsed().as_secs_f64());

    let arms: Vec<(&str, Vec<(&str, &str)>)> = vec![
        ("slab_host_merger", vec![]),
        ("slab_fitted", vec![(DOTS_VISION_WGPU_FITTED_PLANS_ENV, "1")]),
        ("slab_gpu_merger", vec![(DOTS_VISION_WGPU_MERGER_ENV, "gpu")]),
        (
            "slab_fitted_gpu_merger",
            vec![
                (DOTS_VISION_WGPU_FITTED_PLANS_ENV, "1"),
                (DOTS_VISION_WGPU_MERGER_ENV, "gpu"),
            ],
        ),
        (
            "zmerge_fitted_gpu_merger",
            vec![
                (DOTS_VISION_WGPU_FITTED_PLANS_ENV, "1"),
                (DOTS_VISION_WGPU_MERGER_ENV, "gpu"),
                (DOTS_VISION_WGPU_ATTN_ENV, "zmerge"),
            ],
        ),
        (
            "flash_bm64_bn64_sg4",
            vec![
                (DOTS_VISION_WGPU_FITTED_PLANS_ENV, "1"),
                (DOTS_VISION_WGPU_MERGER_ENV, "gpu"),
                (DOTS_VISION_WGPU_ATTN_ENV, "flash"),
                (DOTS_VISION_WGPU_FLASH_PLAN_ENV, "64,64,4,2"),
            ],
        ),
        (
            "flash_bm64_bn64_sg8",
            vec![
                (DOTS_VISION_WGPU_FITTED_PLANS_ENV, "1"),
                (DOTS_VISION_WGPU_MERGER_ENV, "gpu"),
                (DOTS_VISION_WGPU_ATTN_ENV, "flash"),
                (DOTS_VISION_WGPU_FLASH_PLAN_ENV, "64,64,8,2"),
            ],
        ),
        (
            "flash_bm32_bn96_sg4",
            vec![
                (DOTS_VISION_WGPU_FITTED_PLANS_ENV, "1"),
                (DOTS_VISION_WGPU_MERGER_ENV, "gpu"),
                (DOTS_VISION_WGPU_ATTN_ENV, "flash"),
                (DOTS_VISION_WGPU_FLASH_PLAN_ENV, "32,96,4,2"),
            ],
        ),
        (
            "flash_bm128_bn32_sg8",
            vec![
                (DOTS_VISION_WGPU_FITTED_PLANS_ENV, "1"),
                (DOTS_VISION_WGPU_MERGER_ENV, "gpu"),
                (DOTS_VISION_WGPU_ATTN_ENV, "flash"),
                (DOTS_VISION_WGPU_FLASH_PLAN_ENV, "128,32,8,2"),
            ],
        ),
    ];

    let mut baseline: Option<Vec<f32>> = None;
    for round in 0..SWEEP_ROUNDS_ARE_TWO_BECAUSE_A_SIBLING_PROCESS_SHARES_THIS_GPU_AND_ONE_SAMPLE_PER_ARM_CANNOT_NAME_ITS_OWN_NOISE
    {
        for (name, envs) in &arms {
            for (k, v) in envs.iter() {
                std::env::set_var(k, v);
            }
            let t = Instant::now();
            let out = tower
                .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let wall = t.elapsed().as_secs_f64();
            assert!(out.iter().all(|v| v.is_finite()), "{name}: non-finite output");
            let parity = match &baseline {
                Some(b) => max_abs_rel(&out, b),
                None => (0.0, 0.0),
            };
            eprintln!(
                "[dv-sweep] round{round} {name} {wall:.3}s max_abs={:.3e} max_rel={:.3e}",
                parity.0, parity.1
            );
            if baseline.is_none() {
                baseline = Some(out);
            }
            for (k, _) in envs.iter() {
                std::env::remove_var(k);
            }
            tower.apply_m5_fitted_plans(pad_to(n, route.tile()), false);
        }
    }
    let base = baseline.expect("baseline forward ran");

    drop(tower);
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    for wq_name in ["w4g64", "w4g64-twin"] {
        std::env::set_var(DOTS_VISION_WGPU_WQ_ENV, wq_name);
        let wq = VisionWeightFmt::from_env();
        std::env::remove_var(DOTS_VISION_WGPU_WQ_ENV);
        let t1 = Instant::now();
        let tw4 = DotsVisionTowerWgpu::from_loader_route_wq(
            &loader,
            "vision_tower.",
            cfg.clone(),
            route,
            wq,
        )
        .expect("w4 tower load");
        let staged = t1.elapsed().as_secs_f64();
        std::env::set_var(DOTS_VISION_WGPU_FITTED_PLANS_ENV, "1");
        std::env::set_var(DOTS_VISION_WGPU_MERGER_ENV, "gpu");
        let mut walls = Vec::new();
        let mut out = Vec::new();
        for _ in 0..SWEEP_ROUNDS_ARE_TWO_BECAUSE_A_SIBLING_PROCESS_SHARES_THIS_GPU_AND_ONE_SAMPLE_PER_ARM_CANNOT_NAME_ITS_OWN_NOISE
        {
            let t = Instant::now();
            out = tw4
                .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
                .expect("w4 forward");
            walls.push(t.elapsed().as_secs_f64());
        }
        std::env::remove_var(DOTS_VISION_WGPU_FITTED_PLANS_ENV);
        std::env::remove_var(DOTS_VISION_WGPU_MERGER_ENV);
        assert!(out.iter().all(|v| v.is_finite()), "{wq_name}: non-finite output");
        let (mab, mrel) = max_abs_rel(&out, &base);
        eprintln!(
            "[dv-sweep] {wq_name} staged {staged:.1}s, forwards {:?} s, vs f16 slab max_abs={mab:.3e} max_rel={mrel:.3e}",
            walls.iter().map(|w| (w * 1e3).round() / 1e3).collect::<Vec<_>>()
        );
        drop(tw4);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    }

    let (dim, inter) = (cfg.embed_dim, cfg.intermediate_size);
    let hidden = dim * cfg.spatial_merge_size * cfg.spatial_merge_size;
    let per_layer = (dim * 3 + dim) * dim + 2 * inter * dim + dim * inter;
    let quantizable = cfg.num_hidden_layers * per_layer + hidden * hidden + cfg.hidden_size * hidden;
    let f16_bytes = 2 * quantizable;
    let w4_bytes = quantizable / 2 + 2 * 4 * (quantizable / 64);
    eprintln!(
        "[dv-sweep] quantizable weights {quantizable} params: f16 {:.2} GiB, w4g64+scales+biases {:.2} GiB",
        f16_bytes as f64 / (1u64 << 30) as f64,
        w4_bytes as f64 / (1u64 << 30) as f64
    );
}
