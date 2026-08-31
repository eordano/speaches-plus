#![cfg(feature = "wgpu")]

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use nv_kernels::wgpu_backend::WgpuContext;
use nv_models::deepseek_ocr::preprocess::RgbImage;
use nv_models::dots_ocr::preprocess::{prepare, PixelBudget, PATCH_DIM};
use nv_models::dots_ocr::vision::DotsVisionConfig;
use nv_models::dots_ocr::vision_wgpu::{
    pad_to, AttnArm, CoopAcc, CoopPlan, DotsVisionTowerWgpu, VisionGemm, VisionGemmRoute,
    DOTS_VISION_COOP_PLAN_IS_A_SQUARE_FRAGMENT_TILE_BECAUSE_TM_PLUS_TN_OPERAND_LOADS_FEED_TM_TIMES_TN_MULTIPLY_ADDS,
    DOTS_VISION_SPLITK_ENV, DOTS_VISION_STEEL_GEMM_ENV, DOTS_VISION_WGPU_PHASES_ENV,
};
use nv_weights::WeightLoader;

const PAGE_ENV: &str = "NV_DOTS_PAGE";

const COOP_INSTRUMENTS_PIN_STEEL_GEMM_0_BECAUSE_UNSET_NOW_MEANS_THE_STEEL_DEFAULT: &str =
    "Steel gemm is the shipped projection default where the passthrough probe passes, and every \
     row in this file is a claim about the coop kernels, so the towers and ladders here say \
     NV_DOTS_VISION_STEEL_GEMM=0 explicitly; a reference arm that silently becomes the default \
     compares a thing to itself.";

const PATCHES_ENV: &str = "NV_DOTS_VISION_WGPU_PATCHES";

const PIXELS_PER_PATCH_IS_THE_14X14_PATCH_AREA: usize = 196;

const FITTED_PLAN_ENV: &str = "NV_DOTS_VISION_FITTED_PLAN";

const THE_FITTED_ARM_IS_TM8_TN4_EVERYWHERE_THE_FULL_RESOLUTION_LADDER_WON_AND_SHIPPED_WHERE_IT_LOST: &str =
    "A ladder win is a claim about a kernel, not about the tower: it has to survive submit \
     batching, cache state left by the neighbouring phases, and the operand layouts the forward \
     actually uses. This arm installs the winner through the tower's own set_plan and re-times the \
     whole forward, so the phase table is the adjudicator. It lives in the harness and not in \
     resolve_plan because moving the tower's default plan is a performance-default flip.";

fn merged_dim(cfg: &DotsVisionConfig) -> usize {
    cfg.embed_dim * cfg.spatial_merge_size * cfg.spatial_merge_size
}

const THE_PROFILE_DEFAULTS_TO_A_PATCH_COUNT_TWO_FORWARDS_OF_WHICH_FIT_IN_ONE_PROCESS: usize = 3400;

const GEMM_RATE_REPEATS_AMORTIZE_THE_SUBMIT_AND_THE_READBACK_SYNC_OVER_THE_KERNEL: usize = 8;

const THE_FRAGMENT_LOAD_INTENSITY_FORMULA_IS_FITTED_TO_ABOUT_21_FLOP_PER_BYTE_AND_STOPS_PREDICTING_ABOVE_IT_SO_IT_IS_NOT_A_BOUND_TO_REASON_FROM: &str =
    "gemm_coop_f16 stages nothing in workgroup memory: every k step coop-loads tm A fragments and \
     tn B fragments straight from global storage to feed tm*tn multiply-adds, so its fragment-load \
     intensity is 8*tm*tn/(tm+tn) flop per byte on an 8x8 f16 fragment. That expression reproduces \
     this ladder from 16.0 flop per byte at tm4_tn4 to 21.3 at tm8_tn4 AND THEN STOPS: tm8_tn8 \
     accumulating in f16 doubles it to 32.0, fits the lane budget at 64 of 128 dwords, and is \
     SLOWER than tm8_tn4 on every shape measured. So the formula is a curve fitted to the low end \
     of one ladder on one adapter, not a roofline, and whatever binds above roughly 21 flop per \
     byte is UNIDENTIFIED. Do not reason from it: a staged-tile GEMM whose whole premise is that \
     operand reuse is the lever must first establish that operand reuse is still the lever up \
     there, because this ladder is evidence that it is not.";

const A_CALLER_THAT_PRESETS_THE_PHASE_ENV_GETS_ONE_FORWARD_AND_NO_LADDER_BECAUSE_A_SECOND_FULL_RESOLUTION_FORWARD_IN_ONE_PROCESS_DOES_NOT_SURVIVE: &str =
    "One full-resolution forward holds every transient until the next poll, so the untimed control \
     forward and the GEMM ladder that follow it are affordable only below full resolution. Presetting \
     NV_DOTS_VISION_WGPU_PHASES is therefore how a full-page profile asks for the timed forward \
     alone; the drain overhead it costs is measured at 1.05x on the two-forward arm.";

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

fn prepared_page(want: usize) -> nv_models::dots_ocr::preprocess::PreparedImage {
    let page = std::env::var(PAGE_ENV)
        .map(PathBuf::from)
        .expect("NV_DOTS_PAGE must name a page image");
    let img = RgbImage::decode_file(&page).expect("decode page");
    let budget = if want == 0 {
        PixelBudget::from_env()
    } else {
        PixelBudget {
            min_pixels: 3136,
            max_pixels: want * PIXELS_PER_PATCH_IS_THE_14X14_PATCH_AREA,
        }
    };
    prepare(&img, budget).expect("prepare page")
}

fn phase_flops(name: &str, cfg: &DotsVisionConfig, n_pad: usize, tile: usize) -> Option<f64> {
    let (dim, inter, hd) = (cfg.embed_dim, cfg.intermediate_size, cfg.head_dim());
    let (rows, layers, heads) = (n_pad as f64, cfg.num_hidden_layers as f64, cfg.num_attention_heads as f64);
    let mm = |n: usize, k: usize| 2.0 * rows * n as f64 * k as f64;
    Some(match name {
        "patch_embed" => mm(dim, pad_to(PATCH_DIM, tile)),
        "qkv_project" => layers * mm(dim * 3, dim),
        "attn_out_project" => layers * mm(dim, dim),
        "mlp_gate_up_project" => 2.0 * layers * mm(inter, dim),
        "mlp_down_project" => layers * mm(dim, inter),
        "attn_scores_gemm" => layers * heads * mm(n_pad, hd),
        "attn_ctx_gemm" => layers * heads * mm(hd, n_pad),
        "flash_attention" => 2.0 * layers * heads * mm(n_pad, hd),
        _ => return None,
    })
}

const THE_ACC_FRONTIER_IS_TM_TIMES_TN_AGAINST_THE_LANE_DWORD_BUDGET_THE_ACCUMULATOR_DTYPE_SETS: &str =
    "An 8x8 fragment is 64 cells over a 32-lane subgroup, so one f32 accumulator fragment costs 2 \
     lane dwords and one f16 fragment costs 1. Against the 128-dword budget that puts tm4_tn4 at \
     32/128 and 16.0 flop per byte, tm8_tn4 at 64/128 and 21.3, tm8_tn8 at 128/128 and 32.0 with \
     nothing left for operands, and tm8_tn8 accumulating in f16 at 64/128 and 32.0 with half the \
     budget free. The ladder therefore walks tm*tn under both accumulator dtypes at the same sg so \
     the accumulator change is isolated from the tile change.";

fn acc_frontier() -> Vec<(CoopPlan, CoopAcc)> {
    let mut v = Vec::new();
    let _ = THE_ACC_FRONTIER_IS_TM_TIMES_TN_AGAINST_THE_LANE_DWORD_BUDGET_THE_ACCUMULATOR_DTYPE_SETS;
    for acc in [CoopAcc::F32, CoopAcc::F16] {
        for (tm, tn) in [(4u32, 4u32), (8, 4), (4, 8), (8, 8)] {
            v.push((CoopPlan { tm, tn, sg: 4, ku: 1 }, acc));
        }
    }
    v.push((CoopPlan { tm: 4, tn: 4, sg: 4, ku: 1 }, CoopAcc::F32));
    v
}

fn fitted_plan_keys(cfg: &DotsVisionConfig, tile: usize, n_pad: usize) -> Vec<(usize, usize)> {
    let (dim, inter, hd) = (cfg.embed_dim, cfg.intermediate_size, cfg.head_dim());
    vec![
        (dim * 3, dim),
        (dim, dim),
        (inter, dim),
        (dim, inter),
        (dim, pad_to(PATCH_DIM, tile)),
        (merged_dim(cfg), merged_dim(cfg)),
        (cfg.hidden_size, merged_dim(cfg)),
        (n_pad, hd),
    ]
}

#[test]
#[ignore]
fn the_fitted_coop_plan_moves_which_block_owns_a_tile_and_never_the_k_order_so_the_page_is_bit_equal(
) {
    let _ = COOP_INSTRUMENTS_PIN_STEEL_GEMM_0_BECAUSE_UNSET_NOW_MEANS_THE_STEEL_DEFAULT;
    std::env::set_var(DOTS_VISION_STEEL_GEMM_ENV, "0");
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    let tile = route.tile();
    let cfg = DotsVisionConfig::default();
    let prep = prepared_page(THE_PROFILE_DEFAULTS_TO_A_PATCH_COUNT_TWO_FORWARDS_OF_WHICH_FIT_IN_ONE_PROCESS);
    let n_pad = pad_to(prep.num_patches(), tile);
    let dir = snapshot_dir().expect("dots.ocr snapshot present in the hub cache");
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).expect("open dots.ocr weights");
    let tower = DotsVisionTowerWgpu::from_loader_route(&loader, "vision_tower.", cfg.clone(), route)
        .expect("wgpu tower load");

    let shipped = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("shipped-plan forward");
    let fitted_plan = CoopPlan { tm: 8, tn: 4, sg: 4, ku: 1 };
    for (sn, sk) in fitted_plan_keys(&cfg, tile, n_pad) {
        tower.set_plan(sn, sk, Some(fitted_plan));
    }
    let fitted = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("fitted-plan forward");

    assert_eq!(shipped.len(), fitted.len(), "merged feature count moved");
    let differing = shipped
        .iter()
        .zip(fitted.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let mx = shipped.iter().fold(0f32, |a, v| a.max(v.abs()));
    eprintln!(
        "[dots_vision_plan_bits] {} merged features over {} patches, max_abs={mx:.4e}, {differing} \
         differ in a single f32 bit between {} and {}",
        shipped.len(),
        prep.num_patches(),
        DOTS_VISION_COOP_PLAN_IS_A_SQUARE_FRAGMENT_TILE_BECAUSE_TM_PLUS_TN_OPERAND_LOADS_FEED_TM_TIMES_TN_MULTIPLY_ADDS.name(),
        fitted_plan.name()
    );
    assert!(mx > 0.0, "the tower produced an all-zero feature map");
    assert_eq!(
        differing, 0,
        "a coop plan change moves which block owns a tile and never the k order, so any differing \
         bit means the fitted arm is not the same function and cannot be adopted as a free win"
    );
}

const THE_M_ROW_FLASH_IS_PRICED_AGAINST_THE_SCORES_AND_CONTEXT_GEMMS_IT_WOULD_REPLACE_PLUS_THE_SOFTMAX_IT_WOULD_DELETE: &str =
    "The fused arm has to beat attn_scores + attn_ctx, which the phase table reads at 1.755 + 2.388 \
     = 4.143 s for the whole tower under the fitted plan, and it earns the softmax's 1.460 s and \
     attn_merge's 0.033 s for free because it never materialises a score slab. So the bar for one \
     layer-head-pass sweep is 4.143 s of equivalent work, and the arm is worth building only below \
     5.636 s -- above that it is a regression even counting the passes it deletes.";

#[test]
#[ignore]
fn the_m_row_flash_is_priced_at_the_towers_own_attention_shape_across_the_register_cliff() {
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    eprintln!("[dots_vision_flashm] adapter: {}", ctx.summary());
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    assert!(route.is_coop(), "the tower's serving route is the comparison");
    let tile = route.tile();
    let cfg = DotsVisionConfig::default();
    let g = VisionGemm::new(ctx, route).expect("vision gemm");
    let n = std::env::var(PATCHES_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(13600);
    let n_pad = pad_to(n, tile);
    let (heads, hd) = (cfg.num_attention_heads, cfg.head_dim());
    let layers = cfg.num_hidden_layers as f64;
    let _ = THE_M_ROW_FLASH_IS_PRICED_AGAINST_THE_SCORES_AND_CONTEXT_GEMMS_IT_WOULD_REPLACE_PLUS_THE_SOFTMAX_IT_WOULD_DELETE;
    let flops = 2.0 * 2.0 * (n as f64) * (n as f64) * (hd as f64) * (heads as f64);
    eprintln!(
        "[dots_vision_flashm] n={n} n_pad={n_pad} heads={heads} hd={hd}: one pass is {:.1} GFLOP, \
         the tower runs {layers} of them",
        flops / 1e9
    );
    let mut ran = 0usize;
    for m_rows in [8usize, 12, 16] {
        match g.time_flash_m(n, n_pad, heads, hd, m_rows, 3) {
            Ok(secs) => {
                ran += 1;
                eprintln!(
                    "[dots_vision_flashm] M={m_rows} intensity {m_rows} flop/byte, ~{} lane dwords: \
                     {:.2} ms per pass, {:.2} TFLOP/s, {:.3} s over {layers} layers",
                    7 * m_rows,
                    secs * 1e3,
                    flops / secs / 1e12,
                    secs * layers
                );
            }
            Err(why) => eprintln!("[dots_vision_flashm] M={m_rows} did not run: {why}"),
        }
    }
    assert!(
        ran > 0,
        "every M-row arm was dropped, so this table prices nothing -- an arm that never dispatched \
         must fail the probe rather than read as an absent speedup"
    );
}

#[test]
#[ignore]
fn the_f16_accumulator_config_this_adapter_advertises_is_priced_against_the_f32_one_it_has_never_used(
) {
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    eprintln!("[dots_vision_acc] adapter: {}", ctx.summary());
    for c in &ctx.caps.coop_configs {
        eprintln!(
            "[dots_vision_acc] coop config {}x{}x{} ab={} cr={}",
            c.m, c.n, c.k, c.ab, c.cr
        );
    }
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    assert!(route.is_coop(), "the accumulator ladder needs a coop route");
    let _ = COOP_INSTRUMENTS_PIN_STEEL_GEMM_0_BECAUSE_UNSET_NOW_MEANS_THE_STEEL_DEFAULT;
    std::env::set_var(DOTS_VISION_STEEL_GEMM_ENV, "0");
    std::env::set_var(DOTS_VISION_SPLITK_ENV, "0");
    let tile = route.tile();
    let cfg = DotsVisionConfig::default();
    let g = VisionGemm::new(ctx, route).expect("vision gemm");
    let n = std::env::var(PATCHES_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(THE_PROFILE_DEFAULTS_TO_A_PATCH_COUNT_TWO_FORWARDS_OF_WHICH_FIT_IN_ONE_PROCESS);
    let n_pad = pad_to(n, tile);
    let qb = g.attention_query_block_rows(n_pad);
    let merged = merged_dim(&cfg);
    let shapes: Vec<(&str, usize, usize, usize)> = vec![
        ("attn_scores", n_pad, cfg.head_dim(), qb.min(n_pad)),
        ("attn_ctx", cfg.head_dim(), n_pad, qb.min(n_pad)),
        ("qkv", cfg.embed_dim * 3, cfg.embed_dim, n_pad),
        ("attn_proj", cfg.embed_dim, cfg.embed_dim, n_pad),
        ("mlp_fc1_fc3", cfg.intermediate_size, cfg.embed_dim, n_pad),
        ("mlp_fc2", cfg.embed_dim, cfg.intermediate_size, n_pad),
        ("merger_fc0", merged, merged, n_pad / 4),
    ];
    let arms = acc_frontier();
    for (name, sn, sk, sm) in shapes {
        let sm_pad = pad_to(sm, tile);
        let rows = g
            .time_coop_acc_arms(
                name,
                sn,
                sk,
                sm_pad,
                GEMM_RATE_REPEATS_AMORTIZE_THE_SUBMIT_AND_THE_READBACK_SYNC_OVER_THE_KERNEL,
                &arms,
            )
            .expect("accumulator ladder");
        assert!(
            rows.iter().any(|r| r.1 == CoopAcc::F16),
            "{name}: every f16-accumulate arm was dropped, so this table prices nothing new -- an \
             arm that never dispatched must fail the probe, not read as an absent speedup"
        );
        let best = rows.iter().fold(f64::INFINITY, |a, r| a.min(r.2)).max(1e-12);
        for (plan, acc, secs) in &rows {
            eprintln!(
                "[dots_vision_acc] {name} m={sm_pad} n={sn} k={sk} {} {} {:.4} ms {:.2} TFLOP/s \
                 {} lane dwords {:.2}x-best",
                plan.name(),
                acc.name(),
                secs * 1e3,
                2.0 * sm_pad as f64 * sn as f64 * sk as f64 / secs / 1e12,
                acc.lane_dwords(tile as u32, plan.tm, plan.tn),
                secs / best
            );
        }
    }
}

#[test]
#[ignore]
fn the_vision_towers_phase_breakdown_names_which_phase_is_the_wall_on_this_adapter() {
    let _ = COOP_INSTRUMENTS_PIN_STEEL_GEMM_0_BECAUSE_UNSET_NOW_MEANS_THE_STEEL_DEFAULT;
    std::env::set_var(DOTS_VISION_STEEL_GEMM_ENV, "0");
    let ctx = WgpuContext::shared().expect("wgpu adapter");
    eprintln!("[dots_vision_phase] adapter: {}", ctx.summary());
    let route = VisionGemmRoute::from_env_and_adapter(ctx);
    let tile = route.tile();
    let cfg = DotsVisionConfig::default();
    let want = std::env::var(PATCHES_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(THE_PROFILE_DEFAULTS_TO_A_PATCH_COUNT_TWO_FORWARDS_OF_WHICH_FIT_IN_ONE_PROCESS);
    let prep = prepared_page(want);
    let n = prep.num_patches();
    let n_pad = pad_to(n, tile);
    eprintln!(
        "[dots_vision_phase] route={} tile={tile} grid {}x{} = {n} patches (n_pad={n_pad}), attn arm={}",
        route.name(),
        prep.grid_h,
        prep.grid_w,
        AttnArm::from_env().name()
    );

    let dir = snapshot_dir().expect("dots.ocr snapshot present in the hub cache");
    let loader = WeightLoader::open_dir(&dir, &Device::Cpu).expect("open dots.ocr weights");
    let t0 = Instant::now();
    let tower = DotsVisionTowerWgpu::from_loader_route(&loader, "vision_tower.", cfg.clone(), route)
        .expect("wgpu tower load");
    eprintln!(
        "[dots_vision_phase] weight staging {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    if std::env::var(FITTED_PLAN_ENV).is_ok() {
        let _ = THE_FITTED_ARM_IS_TM8_TN4_EVERYWHERE_THE_FULL_RESOLUTION_LADDER_WON_AND_SHIPPED_WHERE_IT_LOST;
        let fitted = CoopPlan { tm: 8, tn: 4, sg: 4, ku: 1 };
        let (dim, inter, hd) = (cfg.embed_dim, cfg.intermediate_size, cfg.head_dim());
        for (sn, sk) in [
            (dim * 3, dim),
            (dim, dim),
            (inter, dim),
            (dim, inter),
            (dim, pad_to(PATCH_DIM, tile)),
            (merged_dim(&cfg), merged_dim(&cfg)),
            (cfg.hidden_size, merged_dim(&cfg)),
            (n_pad, hd),
        ] {
            tower.set_plan(sn, sk, Some(fitted));
        }
        eprintln!(
            "[dots_vision_phase] fitted arm: {} on every shape but attn_ctx (n={hd} k={n_pad}), \
             where the full-resolution ladder reads 0.91x and the shipped plan stays",
            fitted.name()
        );
    }

    let _ = A_CALLER_THAT_PRESETS_THE_PHASE_ENV_GETS_ONE_FORWARD_AND_NO_LADDER_BECAUSE_A_SECOND_FULL_RESOLUTION_FORWARD_IN_ONE_PROCESS_DOES_NOT_SURVIVE;
    let one_forward = std::env::var(DOTS_VISION_WGPU_PHASES_ENV).is_ok();
    let tokens = prep.num_vision_tokens() as f64;
    let mut untimed = f64::NAN;
    if !one_forward {
        let t1 = Instant::now();
        let plain = tower
            .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
            .expect("untimed forward");
        untimed = t1.elapsed().as_secs_f64();
        assert_eq!(
            plain.len(),
            prep.num_vision_tokens() * cfg.hidden_size,
            "merged feature count"
        );
        assert!(plain.iter().all(|v| v.is_finite()));
        eprintln!(
            "[dots_vision_phase] untimed forward {untimed:.3}s for {n} patches, {:.3} ms per vision token",
            untimed * 1e3 / tokens
        );
        std::env::set_var(DOTS_VISION_WGPU_PHASES_ENV, "1");
    }
    let t2 = Instant::now();
    let timed = tower
        .forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
        .expect("phase timed forward");
    let drained = t2.elapsed().as_secs_f64();
    assert_eq!(
        timed.len(),
        prep.num_vision_tokens() * cfg.hidden_size,
        "merged feature count"
    );
    assert!(timed.iter().all(|v| v.is_finite()));
    eprintln!(
        "[dots_vision_phase] drained forward {drained:.3}s ({:.3} ms per vision token) vs untimed \
         {untimed:.3}s, drain overhead {:.2}x",
        drained * 1e3 / tokens,
        drained / untimed.max(1e-9)
    );
    std::env::remove_var(DOTS_VISION_WGPU_PHASES_ENV);

    for name in [
        "patch_embed",
        "qkv_project",
        "attn_out_project",
        "mlp_gate_up_project",
        "mlp_down_project",
        "attn_scores_gemm",
        "attn_ctx_gemm",
        "flash_attention",
    ] {
        if let Some(f) = phase_flops(name, &cfg, n_pad, tile) {
            eprintln!("[dots_vision_phase] {name} costs {:.1} GFLOP per forward", f / 1e9);
        }
    }
    if one_forward {
        return;
    }

    std::env::set_var(DOTS_VISION_SPLITK_ENV, "0");
    let g = VisionGemm::new(ctx, route).expect("vision gemm");
    let qb = g.attention_query_block_rows(n_pad);
    let merged = merged_dim(&cfg);
    let shapes: Vec<(&str, usize, usize, usize)> = vec![
        ("attn_scores", n_pad, cfg.head_dim(), qb.min(n_pad)),
        ("attn_ctx", cfg.head_dim(), n_pad, qb.min(n_pad)),
        ("qkv", cfg.embed_dim * 3, cfg.embed_dim, n_pad),
        ("attn_proj", cfg.embed_dim, cfg.embed_dim, n_pad),
        ("mlp_fc1_fc3", cfg.intermediate_size, cfg.embed_dim, n_pad),
        ("mlp_fc2", cfg.embed_dim, cfg.intermediate_size, n_pad),
        ("patch_proj", cfg.embed_dim, pad_to(PATCH_DIM, tile), n_pad),
        ("merger_fc0", merged, merged, n / (merged / cfg.embed_dim)),
        ("merger_fc2", cfg.hidden_size, merged, n / (merged / cfg.embed_dim)),
    ];
    let _ = THE_FRAGMENT_LOAD_INTENSITY_FORMULA_IS_FITTED_TO_ABOUT_21_FLOP_PER_BYTE_AND_STOPS_PREDICTING_ABOVE_IT_SO_IT_IS_NOT_A_BOUND_TO_REASON_FROM;
    let mut arms = vec![
        DOTS_VISION_COOP_PLAN_IS_A_SQUARE_FRAGMENT_TILE_BECAUSE_TM_PLUS_TN_OPERAND_LOADS_FEED_TM_TIMES_TN_MULTIPLY_ADDS,
    ];
    for (tm, tn) in [(8u32, 8u32), (8, 4), (4, 8), (8, 2), (2, 8), (16, 4), (4, 16)] {
        for sg in [2u32, 4] {
            arms.push(CoopPlan { tm, tn, sg, ku: 1 });
        }
    }
    for (tm, tn, sg, ku) in [(8u32, 8u32, 4u32, 2u32), (4, 4, 4, 2), (8, 4, 4, 2)] {
        arms.push(CoopPlan { tm, tn, sg, ku });
    }
    arms.push(
        DOTS_VISION_COOP_PLAN_IS_A_SQUARE_FRAGMENT_TILE_BECAUSE_TM_PLUS_TN_OPERAND_LOADS_FEED_TM_TIMES_TN_MULTIPLY_ADDS,
    );
    for (name, sn, sk, sm) in shapes {
        let sm_pad = pad_to(sm, tile);
        let rows = g
            .time_gemm_plans(
                name,
                sn,
                sk,
                sm_pad,
                GEMM_RATE_REPEATS_AMORTIZE_THE_SUBMIT_AND_THE_READBACK_SYNC_OVER_THE_KERNEL,
                &arms,
            )
            .expect("shipped-plan gemm rate");
        for (plan, secs) in &rows {
            eprintln!(
                "[dots_vision_phase] gemm {name} m={sm_pad} n={sn} k={sk} {} {:.4} ms {:.2} TFLOP/s",
                plan.name(),
                secs * 1e3,
                2.0 * sm_pad as f64 * sn as f64 * sk as f64 / secs / 1e12
            );
        }
    }
}
