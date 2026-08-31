#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

pub const WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM: usize = 200;
const PLATEAU_WINDOW_4_STEPS_MEDIAN_VS_THE_PREVIOUS_4: usize = 4;
const PLATEAU_FLOOR_8_STEPS_THE_OLD_FIXED_WARMUP_THAT_DID_NOT_RAMP_CLOCKS: usize = 8;
const PLATEAU_RATIO_LAST_WINDOW_MEDIAN_WITHIN_10_PCT_OF_PREVIOUS: f64 = 0.10;
const PLATEAU_CAP_15_SECS_SO_A_SLOW_DEEP_RUNG_CANNOT_WARM_FOREVER: f64 = 15.0;

pub fn serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value(
) -> MutexGuard<'static, ()> {
    static ONE_GPU_TEST_AT_A_TIME: OnceLock<Mutex<()>> = OnceLock::new();
    ONE_GPU_TEST_AT_A_TIME
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned_by_a_sibling_test_panic| poisoned_by_a_sibling_test_panic.into_inner())
}

fn median_of_window(window: &[f64]) -> f64 {
    let mut sorted = window.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
}

pub fn warmup_until_step_time_plateaus_because_post_idle_clock_ramp_reads_2x_slow(
    one_untimed_decode_step: &mut dyn FnMut(),
) -> usize {
    let started = Instant::now();
    let mut ms: Vec<f64> = Vec::new();
    loop {
        let t0 = Instant::now();
        one_untimed_decode_step();
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
        let n = ms.len();
        if n >= PLATEAU_FLOOR_8_STEPS_THE_OLD_FIXED_WARMUP_THAT_DID_NOT_RAMP_CLOCKS
            && n >= 2 * PLATEAU_WINDOW_4_STEPS_MEDIAN_VS_THE_PREVIOUS_4
        {
            let w = PLATEAU_WINDOW_4_STEPS_MEDIAN_VS_THE_PREVIOUS_4;
            let last = median_of_window(&ms[n - w..]);
            let prev = median_of_window(&ms[n - 2 * w..n - w]);
            if prev > 0.0
                && (last - prev).abs()
                    <= PLATEAU_RATIO_LAST_WINDOW_MEDIAN_WITHIN_10_PCT_OF_PREVIOUS * prev
            {
                return n;
            }
        }
        if n >= WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
            || started.elapsed().as_secs_f64()
                >= PLATEAU_CAP_15_SECS_SO_A_SLOW_DEEP_RUNG_CANNOT_WARM_FOREVER
        {
            return n;
        }
    }
}

pub fn warmup_to_plateau_then_time_steps(
    mut one_decode_step: impl FnMut(),
    timed_steps: usize,
) -> (usize, Vec<f64>) {
    let warmup_steps =
        warmup_until_step_time_plateaus_because_post_idle_clock_ramp_reads_2x_slow(
            &mut one_decode_step,
        );
    let mut step_ms = Vec::with_capacity(timed_steps);
    for _ in 0..timed_steps {
        let t0 = Instant::now();
        one_decode_step();
        step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    (warmup_steps, step_ms)
}
pub fn nv_ctx_tokens_env_with_k_suffix() -> Option<Vec<usize>> {
    std::env::var("NV_CTX_TOKENS").ok().map(|v| {
        v.split(',')
            .map(|s| {
                let s = s.trim();
                let (num, mult) = match s.strip_suffix(['k', 'K']) {
                    Some(n) => (n, 1024usize),
                    None => (s, 1usize),
                };
                num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
            })
            .collect()
    })
}

pub fn ctx_tokens_from_env_default_256_8k_120k_because_max_pos_131072_has_no_room_for_168k(
) -> Vec<usize> {
    nv_ctx_tokens_env_with_k_suffix().unwrap_or_else(|| vec![256, 8 * 1024, 120 * 1024])
}

pub fn median_ms_of_timed_steps(step_ms: &mut Vec<f64>) -> f64 {
    step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    step_ms[step_ms.len() / 2]
}

pub fn sorted_median_p10_p90_asserting_a_positive_finite_median(step_ms: &[f64]) -> (f64, f64, f64) {
    let mut sorted = step_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |q: f64| {
        let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };
    let (median, p10, p90) = (pick(0.5), pick(0.1), pick(0.9));
    assert!(
        median.is_finite() && median > 0.0,
        "median decode step time is not a positive number: {median}"
    );
    (median, p10, p90)
}

const CANONICAL_LINE_BATCH_1_EVERY_FLEET_CTX_INSTRUMENT_DECODES_ONE_SEQUENCE: usize = 1;

pub struct NvMeasureCtxLine<'a> {
    pub instrument: &'a str,
    pub model_dir: &'a std::path::Path,
    pub backend: &'a str,
    pub device: String,
    pub tokens_primed_kv_depth_of_this_run_never_a_capacity: usize,
    pub steps: usize,
    pub warmup: usize,
    pub value: f64,
    pub unit: &'a str,
    pub basis: &'a str,
}

pub fn emit_nv_measure_v1_alongside_the_legacy_line_which_stays_for_one_release(
    line: NvMeasureCtxLine<'_>,
    extras: &[(&str, String)],
) {
    let mut m = nv_config::measure::Measurement {
        instrument: line.instrument.to_string(),
        model_at_rev: nv_config::measure::model_at_rev_owner_slash_repo_at_short_rev_from_the_hub_snapshot_dir_the_instrument_already_resolves(line.model_dir),
        backend: line.backend.to_string(),
        device: line.device,
        batch: CANONICAL_LINE_BATCH_1_EVERY_FLEET_CTX_INSTRUMENT_DECODES_ONE_SEQUENCE,
        tokens: line.tokens_primed_kv_depth_of_this_run_never_a_capacity,
        steps: line.steps,
        warmup: line.warmup,
        value: line.value,
        unit: line.unit.to_string(),
        extras: Vec::new(),
    };
    m = m.extra("basis", line.basis);
    for (k, v) in extras {
        m = m.extra(*k, v.clone());
    }
    m.emit();
}

#[cfg(feature = "cuda")]
pub fn cuda_device_name_for_the_nv_measure_device_basis_field() -> String {
    cudarc::driver::CudaContext::new(0)
        .ok()
        .and_then(|c| c.name().ok())
        .map(|n| format!("{n} (CUDA)"))
        .unwrap_or_else(|| "unresolved-cuda-device".to_string())
}

#[cfg(feature = "wgpu")]
pub fn wgpu_device_name_for_the_nv_measure_device_basis_field() -> String {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            let info = ctx.adapter.get_info();
            format!("{} ({:?})", info.name, info.backend)
        }
        Err(_) => "unresolved-wgpu-adapter".to_string(),
    }
}

#[cfg(feature = "cuda")]
pub const Q36_Q38_DECODE_NUMERICS_GATES_EVERY_ONE_APPEARS_RESOLVED_IN_THE_ARM_LABEL: [&str; 11] = [
    "NV_Q36_GRAPHED_DECODE_FIX",
    "NV_Q36_GDN_FP8",
    "NV_Q36_LM_HEAD_FP8",
    "NV_Q38_FP4_MLP",
    "NV_Q38_GDN_SPLIT",
    "NV_Q38_MLP_W4A8",
    "NV_Q38_W4A8_MK",
    "NV_Q38_SILUQ_SPLIT",
    "NV_Q38_NORM_QUANT_FOLD",
    "NV_Q38_NORM_WRITEOUT",
    "NV_NVFP4_M1_GEMV",
];

#[cfg(feature = "cuda")]
pub const LAGUNA_DECODE_NUMERICS_GATES_EVERY_ONE_APPEARS_RESOLVED_IN_THE_ARM_LABEL: [&str; 12] = [
    "NV_MOE_SHARED_FOLD",
    "NV_MOE_ROUTE_FUSE",
    "NV_MOE_CHAIN_SLIM",
    "NV_MOE_FP4_DECODE_GEMV",
    "NV_MOE_FP4_DECODE_TILE",
    "NV_LAGUNA_ATTN_FP8",
    "NV_LAGUNA_ATTN_W8",
    "NV_LAGUNA_M1_DEPTHGATE",
    "NV_LAGUNA_M1_FLASH",
    "NV_LAGUNA_M1_QKV",
    "NV_LAGUNA_M1_QKV_Q8",
    "NV_LAGUNA_MOE_NORMGATE_FUSE",
];

#[cfg(feature = "cuda")]
pub const GEMMA4_DECODE_NUMERICS_GATES_EVERY_ONE_APPEARS_RESOLVED_IN_THE_ARM_LABEL: [&str; 11] = [
    "NV_ATTN_PROJ_QUANT",
    "NV_NVFP4_M1_GEMV",
    "NV_NVFP4_TRUE_M",
    "NV_G4_LMHEAD_I8",
    "NV_VERIFY_LMHEAD_INT8",
    "NV_LINEAR_BF16_FORCE_CUBLAS",
    "NV_FP8_SCALE_MODE",
    "NV_GEMMA4_FA2_FULL_DECODE",
    "NV_KV_RING",
    "NV_KV_NO_SLIDING",
    "NV_PREFILL_W4A4",
];

#[cfg(feature = "cuda")]
pub fn arm_label_from_the_resolved_value_of_every_gate_accessor_because_a_presence_label_lies_after_a_default_flip(
    keys: &[&str],
) -> String {
    keys.iter()
        .map(|k| {
            format!(
                "{k}={}",
                resolved_gate_value_from_the_accessor_the_kernel_launcher_reads(k)
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(feature = "cuda")]
fn moe_fp4_decode_tile_opt_in_mirrors_the_private_use_decode_tile_env_read() -> bool {
    matches!(
        std::env::var("NV_MOE_FP4_DECODE_TILE").ok().as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

#[cfg(feature = "cuda")]
fn laguna_normgate_fuse_opt_in_mirrors_the_private_normgate_fuse_enabled_env_read() -> bool {
    std::env::var("NV_LAGUNA_MOE_NORMGATE_FUSE")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn laguna_m1_depthgate_default_on_mirrors_the_laguna_step_cu_env_read_off_only_when_the_value_starts_with_0(
) -> bool {
    std::env::var("NV_LAGUNA_M1_DEPTHGATE")
        .map(|v| !v.starts_with('0'))
        .unwrap_or(true)
}

#[cfg(feature = "cuda")]
fn resolved_gate_value_from_the_accessor_the_kernel_launcher_reads(key: &str) -> String {
    fn on_off(on: bool) -> String {
        if on {
            "on".to_string()
        } else {
            "off".to_string()
        }
    }
    match key {
        "NV_Q36_GRAPHED_DECODE_FIX" => on_off(
            nv_models::qwen3_5_moe::nv_q36_graphed_decode_fix_env_routes_decode_to_the_gemma4_proven_splitk_flash(),
        ),
        "NV_Q36_GDN_FP8" => on_off(
            nv_models::qwen3_5_moe::nv_q36_gdn_fp8_env_opt_in_quantizes_bf16_checkpoint_gdn_projections_to_fp8_resident(),
        ),
        "NV_Q36_LM_HEAD_FP8" => on_off(
            nv_models::qwen3_5_moe::nv_q36_lm_head_fp8_env_opt_in_quantizes_a_bf16_checkpoint_lm_head_to_fp8_resident(),
        ),
        "NV_Q38_FP4_MLP" => on_off(
            nv_models::qwen3_5_moe::nv_q38_fp4_mlp_gemv_env_opt_in_nv_q38_fp4_mlp_1_because_w4a16_matches_but_does_not_beat_the_padded_tc_route_in_graph(),
        ),
        "NV_Q38_GDN_SPLIT" => on_off(
            nv_layers::linear_attn::gdn_split_env_read_per_call_so_the_kill_switch_works_mid_process(),
        ),
        "NV_Q38_MLP_W4A8" => on_off(
            nv_models::qwen3_5_moe::nv_q38_mlp_w4a8_env_opt_in_nv_q38_mlp_w4a8_1_dp4a_int8_dot_route_for_the_nvfp4_dense_mlp_decode(),
        ),
        "NV_Q38_W4A8_MK" => on_off(
            nv_models::qwen3_5_moe::nv_q38_w4a8_mk_env_opt_in_nv_q38_w4a8_mk_1_small_m_dp4a_verify_route(),
        ),
        "NV_Q38_SILUQ_SPLIT" => on_off(
            nv_models::qwen3_5_moe::nv_q38_siluq_split_env_opt_in_nv_q38_siluq_split_1_multiblock_silu_quant_producer(),
        ),
        "NV_Q38_NORM_QUANT_FOLD" => on_off(
            nv_models::qwen3_5_moe::nv_q38_norm_quant_fold_env_opt_in_nv_q38_norm_quant_fold_1_post_norm_residual_rowquant_one_kernel(),
        ),
        "NV_Q38_NORM_WRITEOUT" => on_off(
            nv_models::qwen3_5_moe::nv_q38_norm_writeout_env_opt_in_nv_q38_norm_writeout_1(),
        ),
        "NV_NVFP4_M1_GEMV" => {
            on_off(nv_config::env_arm_opt_in_eq_some_1_reread_every_call(key))
        }
        "NV_MOE_SHARED_FOLD" => on_off(nv_layers::moe_grouped::folded_shared_enabled()),
        "NV_MOE_ROUTE_FUSE" => on_off(nv_layers::moe_grouped::route_fuse_enabled()),
        "NV_MOE_CHAIN_SLIM" => on_off(nv_layers::moe_grouped::chain_slim_launches_enabled()),
        "NV_MOE_FP4_DECODE_GEMV" => on_off(
            nv_layers::moe_grouped::moe_arm_on_unless_the_key_is_explicitly_off_which_is_how_the_pre_round_82_path_is_reached(key),
        ),
        "NV_MOE_FP4_DECODE_TILE" => {
            on_off(moe_fp4_decode_tile_opt_in_mirrors_the_private_use_decode_tile_env_read())
        }
        "NV_LAGUNA_ATTN_FP8" => on_off(
            nv_models::laguna::attn_fp8_default_on_since_round_82_set_the_key_to_0_for_the_pre_round_82_bf16_projections(),
        ),
        "NV_LAGUNA_ATTN_W8" => on_off(
            nv_models::laguna::attn_w8_int8_arm_was_chosen_explicitly_and_therefore_outranks_the_fp8_default(),
        ),
        "NV_LAGUNA_M1_FLASH" | "NV_LAGUNA_M1_QKV" | "NV_LAGUNA_M1_QKV_Q8" => on_off(
            nv_config::env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(key),
        ),
        "NV_LAGUNA_MOE_NORMGATE_FUSE" => {
            on_off(laguna_normgate_fuse_opt_in_mirrors_the_private_normgate_fuse_enabled_env_read())
        }
        "NV_LAGUNA_M1_DEPTHGATE" => on_off(
            laguna_m1_depthgate_default_on_mirrors_the_laguna_step_cu_env_read_off_only_when_the_value_starts_with_0(),
        ),
        "NV_ATTN_PROJ_QUANT" => {
            std::env::var(key).unwrap_or_else(|_| "off".to_string())
        }
        "NV_NVFP4_TRUE_M" => on_off(nv_layers::linear::nvfp4_true_m_from(
            std::env::var(key).ok().as_deref(),
        )),
        "NV_G4_LMHEAD_I8" => {
            on_off(nv_config::env_arm_opt_in_eq_some_1_reread_every_call(key))
        }
        "NV_VERIFY_LMHEAD_INT8" => on_off(
            nv_config::env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(key),
        ),
        "NV_LINEAR_BF16_FORCE_CUBLAS" | "NV_GEMMA4_FA2_FULL_DECODE" | "NV_KV_NO_SLIDING" => {
            on_off(nv_config::env_arm_present_at_all_reread_every_call(key))
        }
        "NV_FP8_SCALE_MODE" => format!("{:?}", nv_layers::linear::fp8_scale_mode()),
        "NV_KV_RING" => on_off(nv_models::gemma4::kv_ring_enabled()),
        "NV_PREFILL_W4A4" => on_off(nv_models::gemma4::prefill_w4a4_env(
            std::env::var(key).ok().as_deref(),
        )),
        other => panic!(
            "{other} has no resolved-gate mapping; wire its accessor here -- a label built from \
             env presence lies the moment the gate's default flips"
        ),
    }
}

pub fn skip_when_env_gate_not_1_printing_the_exact_skip_line(gate_env: &str) -> bool {
    if std::env::var(gate_env).ok().as_deref() != Some("1") {
        eprintln!("skip: {gate_env} != 1");
        return true;
    }
    false
}

#[cfg(feature = "wgpu")]
pub fn print_wgpu_profile_share_table_then_reset(tag: &str) {
    use nv_kernels::wgpu_backend::dispatch::profile;
    if !profile::enabled() {
        return;
    }
    let rows = profile::report();
    let n: u64 = rows.iter().map(|r| r.1).sum();
    let total: f64 = rows.iter().map(|r| r.2).sum();
    eprintln!(
        "[wgpu-prof] {tag}: {} labels {n} dispatches {:.1} ms GPU",
        rows.len(),
        total / 1e6
    );
    for (label, count, ns) in rows.into_iter().take(48) {
        eprintln!(
            "[wgpu-prof] {tag} | {label} n={count} total_ms={:.1} share={:.1}%",
            ns / 1e6,
            100.0 * ns / total.max(1.0)
        );
    }
    profile::reset();
}
