#![cfg(feature = "wgpu")]

mod common;
use common::gow_tiny_weights as tiny_weights;
use common::tiny_config_gpt_oss as tiny_config;
use nv_models::gpt_oss_wgpu as gow;

const ARM_ENVS: [&str; 13] = [
    gow::FLASH_DECODE_ENV,
    gow::FLASH_SLIDING_ENV,
    gow::FUSED_STEP_ENV,
    gow::GEMV_SG_ENV,
    gow::MX_SG_GATE_ENV,
    gow::MX_SG_XV4_ENV,
    gow::MX_SG_BYTE_LUT_ENV,
    gow::MX_SG_XSTAGE_ENV,
    gow::MX_SG_GUPAIR_ENV,
    gow::MX_SG_GUSW_ENV,
    gow::MOE_DNCOMB_FUSE_ENV,
    gow::MOE_DNCOMB_SGSLOT_ENV,
    gow::FLASH_LANEPOS_ENV,
];

#[test]
fn the_arm_stack_ships_on_engages_for_real_and_the_zero_escape_restores_every_serial_arm() {
    assert!(
        gow::GOW_ARM_STACK_IS_THE_DEFAULT_FLASH_SLIDING_FUSED_GEMV_SG_AND_MX_SG_TOGETHER_CUT_KV8_DECODE_7P9_TO_6P5_MS_AT_256_AND_8P3_TO_7P0_AT_8K_PPL_65P7_IN_THE_58_TO_92_BAND_AND_ZERO_RESTORES_EACH_SERIAL_ARM,
        "the arm stack ships on: the quiet-window kv8-x-arms cells and the arm-stack ppl in \
         perf/runs.jsonl are the record, and each NV_GPTOSS_WGPU_* env at 0 restores its \
         serial arm"
    );
    assert!(
        gow::GOW_MOE_FUSE_STACK_XV4_LUT_XSTAGE_GUPAIR_GUSW_DNCOMB_SGSLOT_SHIPS_ON_ROWS_GPTOSS_FUSE_GATED_LADDER_175P7_166P8_107P6_TOK_S_AT_256_8K_122K_CAMPAIGN_PLUS_19P9_35P6_58P2_PCT_AND_ZERO_RESTORES_EACH_ARM,
        "the champion 7-env MoE fuse stack ships on: rows build_name \"gptoss fuse\" in \
         perf/runs.jsonl are the record, and each NV_GOW_* env at 0 restores its serial arm"
    );
    assert!(
        gow::LANEPOS_GATED_ROUTE_SHIPS_ON_ROWS_GPTOSS_FUSE_GATED_LADDER_175P7_166P8_107P6_TOK_S_AT_256_8K_122K_SHALLOW_ARGMAX_AB_FAILS_BELOW_THE_KNEE_SO_THE_GATE_NEVER_EXPOSES_IT_AND_ZERO_RESTORES_FDF_ONLY,
        "the lanepos gated route ships on: rows build_name \"gptoss fuse\" (gated ladder) are \
         the record, and NV_GOW_FLASH_LANEPOS=0 restores the fdf-only route"
    );
    let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("[wgpu] adapter: {}", ctx.info.name);
            ctx
        }
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            return;
        }
    };
    for env in ARM_ENVS {
        assert!(
            std::env::var(env).is_err(),
            "{env} leaked into the default-contract probe; this test must see the shipped \
             defaults, not a caller's pins"
        );
    }
    assert_eq!(
        gow::flash_lanepos_mode_unset_gates_at_the_measured_knee_1_always_0_restores_fdf_only(),
        gow::FlashLaneposMode::Gate(
            gow::LANEPOS_ENGAGES_AT_THE_MEASURED_8192_CROSSOVER_FDF_WINS_SHALLOW_LANEPOS_FROM_16K
        ),
        "unset {} must ship the gated lanepos route at the measured 8192 knee",
        gow::FLASH_LANEPOS_ENV
    );
    std::env::set_var(gow::FLASH_LANEPOS_ENV, "0");
    assert_eq!(
        gow::flash_lanepos_mode_unset_gates_at_the_measured_knee_1_always_0_restores_fdf_only(),
        gow::FlashLaneposMode::Off,
        "{}=0 must restore the fdf-only route",
        gow::FLASH_LANEPOS_ENV
    );
    std::env::set_var(gow::FLASH_LANEPOS_ENV, "1");
    assert_eq!(
        gow::flash_lanepos_mode_unset_gates_at_the_measured_knee_1_always_0_restores_fdf_only(),
        gow::FlashLaneposMode::Always,
        "{}=1 must keep forcing lanepos at every depth",
        gow::FLASH_LANEPOS_ENV
    );
    std::env::set_var(gow::FLASH_LANEPOS_ENV, "gate");
    assert_eq!(
        gow::flash_lanepos_mode_unset_gates_at_the_measured_knee_1_always_0_restores_fdf_only(),
        gow::FlashLaneposMode::Gate(
            gow::LANEPOS_ENGAGES_AT_THE_MEASURED_8192_CROSSOVER_FDF_WINS_SHALLOW_LANEPOS_FROM_16K
        ),
        "{}=gate must keep its explicit gated meaning",
        gow::FLASH_LANEPOS_ENV
    );
    std::env::remove_var(gow::FLASH_LANEPOS_ENV);
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x60F1_A555);
    let mut shipped = gow::GptOssWgpu::new(cfg.clone(), &hw, 32).expect("build shipped-default model");
    let labels = shipped.decode_pass_labels().join(" ");
    assert!(
        labels.contains("gow-at-flash1"),
        "flash decode did not engage on the shipped default (labels: {labels})"
    );
    assert!(
        labels.contains("gow-at-furopekv"),
        "the fused qkv+rope step did not engage on the shipped default (labels: {labels})"
    );
    assert!(
        !labels.contains("gow-at-decode:"),
        "the serial attention arm is still dispatched under the shipped default, so the flip \
         silently fell back (labels: {labels})"
    );
    if nv_kernels::wgpu_backend::kernels::gemv_nvfp4::sg32_ok(ctx) {
        assert_eq!(
            shipped.mx_gemv_entry(),
            gow::MX_SG_XV4_R2_LUT_XS_ENTRY,
            "the champion xv4+lut+xstage+gupair mx arm must be the shipped default on a \
             uniformly-32-wide-subgroup adapter"
        );
        assert!(
            labels.contains(gow::MX_SG_XV4_R4_LUT_XS_SWIGLU_ENTRY),
            "the fused gu+swiglu arm did not engage on the shipped default (labels: {labels})"
        );
        assert!(
            !labels.contains(":gow_swiglu"),
            "the split swiglu pass is still dispatched under the shipped gusw default \
             (labels: {labels})"
        );
        assert!(
            labels.contains("gow-moe-gemv-dncomb:"),
            "the dncomb fused down+combine arm did not engage on the shipped default \
             (labels: {labels})"
        );
        assert!(
            !labels.contains("gow-moe-combine:"),
            "the split combine pass is still dispatched under the shipped dncomb default \
             (labels: {labels})"
        );
    } else {
        eprintln!("[skip] subgroup width is not uniformly 32; the sg arms clamp to scalar here");
    }
    let refusal = shipped
        .verify_chain(&[1])
        .expect_err("the verify surface must refuse the arm-stack default, not serve rows from it");
    assert!(
        format!("{refusal:#}").contains("=0"),
        "the verify refusal must name the escape envs, got: {refusal:#}"
    );

    for env in ARM_ENVS {
        std::env::set_var(env, "0");
    }
    let serial_build = gow::GptOssWgpu::new(cfg.clone(), &hw, 32);
    for env in ARM_ENVS {
        std::env::remove_var(env);
    }
    let serial = serial_build.expect("build serial-escape model");
    let slabels = serial.decode_pass_labels().join(" ");
    assert!(
        slabels.contains("gow-at-decode:"),
        "the zero escape did not restore the serial attention arm (labels: {slabels})"
    );
    assert!(
        !slabels.contains("gow-at-flash1") && !slabels.contains("gow-at-furopekv"),
        "the zero escape left flash or fused passes in the decode list (labels: {slabels})"
    );
    assert_eq!(
        serial.mx_gemv_entry(),
        gow::MX_SCALAR_ENTRY,
        "the zero escape must restore the audited scalar mx arm"
    );
    assert!(
        slabels.contains(":gow_swiglu") && slabels.contains("gow-moe-combine:"),
        "the zero escape must restore the split swiglu and combine passes (labels: {slabels})"
    );
    assert!(
        !slabels.contains("gow-moe-gemv-dncomb:"),
        "the zero escape left the dncomb fused pass in the decode list (labels: {slabels})"
    );
    assert!(
        shipped.pass_count() < serial.pass_count(),
        "the shipped default must dispatch fewer decode passes than the serial escape \
         ({} vs {}), or the fused/flash arms never engaged",
        shipped.pass_count(),
        serial.pass_count()
    );
}
