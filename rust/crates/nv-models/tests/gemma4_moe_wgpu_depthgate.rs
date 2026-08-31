#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip_no_require as ctx_or_skip;
use common::tensors_for_cfg_two_sided as tensors_for_cfg;
use common::EnvPins;
use common::TempDir;
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu as g4m;
use nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu;
use nv_weights::WeightLoader;

const HIDDEN: usize = 64;
const INTER: usize = 96;
const N_LAYERS: usize = 3;
const N_Q: usize = 4;
const N_KV: usize = 2;
const N_GLOBAL_KV: usize = 1;
const HEAD_DIM: usize = 16;
const GLOBAL_HEAD_DIM_32_ONE_FULL_LANE_STRIP: usize = 32;
const VOCAB: usize = 160;
const N_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const MOE_INTER: usize = 64;

const SLIDING_WINDOW_640_SITS_BETWEEN_KNEE1_AND_KNEE2_SO_THE_SLIDING_BAND_PINS_AT_12_WHILE_FULL_CLIMBS_TO_32:
    usize = 640;

const TINY_MAX_SEQ_1200_CLEARS_KNEE3_AT_1024_WITH_ROOM_FOR_TIMED_STEPS: usize = 1200;

const SHALLOW_SPLITS_8_IS_THE_ROUND_10_RECORD_ARM_PIN_TUNED_AT_256: u32 = 8;

const TINY_PREFILL_M_PINNED: usize = 16;

static ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE:
    std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ARM_ENVS_ARE_PROCESS_GLOBAL_SO_TESTS_MUTATING_THEM_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn tiny_config_json() -> String {
    let window = SLIDING_WINDOW_640_SITS_BETWEEN_KNEE1_AND_KNEE2_SO_THE_SLIDING_BAND_PINS_AT_12_WHILE_FULL_CLIMBS_TO_32;
    format!(
        r#"{{
  "model_type": "gemma4",
  "tie_word_embeddings": true,
  "text_config": {{
    "attention_k_eq_v": true,
    "enable_moe_block": true,
    "final_logit_softcapping": 30.0,
    "global_head_dim": {GLOBAL_HEAD_DIM_32_ONE_FULL_LANE_STRIP},
    "head_dim": {HEAD_DIM},
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": {HIDDEN},
    "intermediate_size": {INTER},
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
    "max_position_embeddings": 2048,
    "moe_intermediate_size": {MOE_INTER},
    "num_attention_heads": {N_Q},
    "num_experts": {N_EXPERTS},
    "num_global_key_value_heads": {N_GLOBAL_KV},
    "num_hidden_layers": {N_LAYERS},
    "num_key_value_heads": {N_KV},
    "rms_norm_eps": 1e-06,
    "rope_parameters": {{
      "full_attention": {{"partial_rotary_factor": 0.25, "rope_theta": 1000000.0}},
      "sliding_attention": {{"rope_theta": 10000.0}}
    }},
    "sliding_window": {window},
    "tie_word_embeddings": true,
    "top_k_experts": {TOP_K},
    "vocab_size": {VOCAB}
  }}
}}"#
    )
}

fn temp_dir(tag: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("g4m_dgate_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

fn tiny_model_gated(
    tag: &str,
    seed: u64,
    max_seq: usize,
    depthgate: Option<&str>,
    foldgate: Option<&str>,
    preenc: Option<&str>,
) -> Gemma4MoeWgpu {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let dir = temp_dir(tag);
    let st = dir.0.join("model.safetensors");
    candle_core::safetensors::save(&tensors_for_cfg(&cfg, seed), &st).unwrap();
    let loader = WeightLoader::open_file(&st, &candle_core::Device::Cpu).unwrap();
    let m_s = TINY_PREFILL_M_PINNED.to_string();
    let pins = EnvPins::pin(&[
        ("NV_G4MOE_FLASH_DECODE", Some("1")),
        ("NV_G4MOE_FLASH_SLIDING", Some("1")),
        ("NV_G4MOE_FLASH_SPLITS", Some("8")),
        (g4m::G4MOE_FLASH_DEPTHGATE_ENV, depthgate),
        (g4m::G4MOE_FLASH_DEPTHGATE_SPLITS_ENV, None),
        (g4m::G4MOE_FLASH_FOLDGATE_ENV, foldgate),
        (g4m::FLASH_GQA_FOLD_ENV, None),
        (g4m::G4MOE_FLASH_SLIDING_FOLD_ENV, None),
        (g4m::G4MOE_FLASH_SLIDING_SPLITS_ENV, None),
        (g4m::FLASH_SACC_CHUNK_ENV, None),
        ("NV_G4MOE_PREENC", preenc),
        ("NV_G4MOE_QKVFOLD", None),
        ("NV_G4MOE_KV_FP8", Some("0")),
        ("NV_G4MOE_KV_RING", Some("0")),
        ("NV_G4MOE_WGPU_PREFILL_M", Some(m_s.as_str())),
        ("NV_WGPU_PREFILL_M", None),
        ("NV_G4MOE_GEMV_WIDE", None),
        ("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED", None),
    ]);
    let m = Gemma4MoeWgpu::from_loader(cfg, &loader, max_seq);
    drop(pins);
    m.unwrap()
}

fn tiny_model(gate: bool, seed: u64) -> Gemma4MoeWgpu {
    tiny_model_gated(
        if gate { "gate" } else { "off" },
        seed,
        TINY_MAX_SEQ_1200_CLEARS_KNEE3_AT_1024_WITH_ROOM_FOR_TIMED_STEPS,
        gate.then_some("1"),
        None,
        None,
    )
}

#[test]
fn every_band_boundary_is_a_constant_the_gate_reads_and_the_ladder_rises_one_then_two_workgroups_per_sm(
) {
    let shallow = SHALLOW_SPLITS_8_IS_THE_ROUND_10_RECORD_ARM_PIN_TUNED_AT_256;
    let knee1 = g4m::FLASH_DEPTHGATE_KNEE_TOKENS_512_IS_WHERE_12_FIRST_BEATS_THE_SHALLOW_PIN_AND_BELOW_IT_8_WINS;
    let knee2 = g4m::FLASH_DEPTHGATE_KNEE2_TOKENS_768_IS_WHERE_24_OVERTAKES_12_ON_THE_SAME_LADDER;
    let knee3 =
        g4m::FLASH_DEPTHGATE_KNEE3_TOKENS_1024_IS_WHERE_32_OVERTAKES_24_AND_WINS_EVERY_DEPTH_BEYOND;
    let band1 =
        g4m::FLASH_DEPTHGATE_BAND1_SPLITS_12_IS_192_WORKGROUPS_ONE_PER_SM_AT_16_Q_HEADS_ON_188_SMS;
    let band2 =
        g4m::FLASH_DEPTHGATE_BAND2_SPLITS_24_IS_384_WORKGROUPS_TWO_PER_SM_AT_16_Q_HEADS_ON_188_SMS;
    let band3 = g4m::FLASH_DEPTHGATE_BAND3_SPLITS_32_IS_512_WORKGROUPS_THE_MEASURED_RESIDENCY_CEILING_AND_47_AND_64_LOSE_ABOVE_IT;
    assert!(
        knee1 < knee2 && knee2 < knee3 && band1 < band2 && band2 < band3,
        "the measured ladder rises 8 -> 12 -> 24 -> 32 as depth grows; unordered bands mean the \
         gate would step down at depth, which is the starvation it exists to fix"
    );
    for tokens in [1u32, 256, knee1 - 1] {
        assert_eq!(
            g4m::flash_depthgate_splits_for_context_tokens(tokens, shallow),
            shallow,
            "below the knee the gate must return the caller's tuned split count unchanged at \
             {tokens} tokens, or the round-10 record arm's shallow pin stops being honoured"
        );
    }
    for (tokens, want) in [
        (knee1, band1),
        (knee2 - 1, band1),
        (knee2, band2),
        (knee3 - 1, band2),
        (knee3, band3),
        (8192, band3),
        (122880, band3),
    ] {
        assert_eq!(
            g4m::flash_depthgate_splits_for_context_tokens(tokens, shallow),
            want,
            "the band at {tokens} tokens moved off the measured ladder"
        );
    }
    assert_eq!(
        g4m::flash_depthgate_splits_for_context_tokens(1, 64),
        64,
        "an explicit split count is the SHALLOW band only; above the knee the measured bands own \
         the grid, which is the whole point of the gate"
    );
    assert_eq!(
        g4m::flash_depthgate_splits_for_context_tokens(122880, 64),
        band3,
        "above the knee a caller pin must not survive: 64 measured 19.703 against 18.900 for the \
         banded 32 at 122880"
    );
}

#[test]
fn the_scratch_provision_covers_every_band_the_gate_can_pick_or_a_deep_step_writes_out_of_bounds() {
    let _g = env_lock();
    std::env::set_var(g4m::G4MOE_FLASH_DEPTHGATE_ENV, "1");
    for shallow in [1u32, 8, 12, 24, 32, 64] {
        let provisioned =
            g4m::flash_depthgate_scratch_splits_must_cover_every_band_the_gate_can_pick(shallow);
        for tokens in [1u32, 512, 768, 1024, 8192, 122880] {
            assert!(
                g4m::flash_depthgate_splits_for_context_tokens(tokens, shallow) <= provisioned,
                "shallow {shallow} at {tokens} tokens picks more splits than the scratch rows \
                 provisioned; stage1 indexes scratch by split and would write past the buffer"
            );
        }
    }
    assert_eq!(
        g4m::flash_depthgate_scratch_splits_must_cover_every_band_the_gate_can_pick(
            SHALLOW_SPLITS_8_IS_THE_ROUND_10_RECORD_ARM_PIN_TUNED_AT_256
        ),
        g4m::FLASH_DEPTHGATE_BAND3_SPLITS_32_IS_512_WORKGROUPS_THE_MEASURED_RESIDENCY_CEILING_AND_47_AND_64_LOSE_ABOVE_IT,
        "the record arm's shallow pin of 8 must still provision the deepest band's scratch rows"
    );
    std::env::remove_var(g4m::G4MOE_FLASH_DEPTHGATE_ENV);
    assert_eq!(
        g4m::flash_depthgate_scratch_splits_must_cover_every_band_the_gate_can_pick(
            SHALLOW_SPLITS_8_IS_THE_ROUND_10_RECORD_ARM_PIN_TUNED_AT_256
        ),
        SHALLOW_SPLITS_8_IS_THE_ROUND_10_RECORD_ARM_PIN_TUNED_AT_256,
        "with the gate off the scratch must not grow: an unarmed build owes zero extra VRAM"
    );
}

#[test]
fn the_gate_is_default_off_and_both_of_its_envs_are_opt_in() {
    let _g = env_lock();
    std::env::remove_var(g4m::G4MOE_FLASH_DEPTHGATE_ENV);
    std::env::remove_var(g4m::G4MOE_FLASH_DEPTHGATE_SPLITS_ENV);
    assert!(
        !g4m::flash_depthgate_enabled_a_shallow_tuned_split_pin_starves_the_grid_at_depth(),
        "the depth gate reassociates the split-k reduction, so it must never be the default \
         until a round flips it deliberately"
    );
    for off in ["0", "", "yes", "true"] {
        std::env::set_var(g4m::G4MOE_FLASH_DEPTHGATE_ENV, off);
        assert!(
            !g4m::flash_depthgate_enabled_a_shallow_tuned_split_pin_starves_the_grid_at_depth(),
            "only the exact string 1 arms the gate; {off:?} must not"
        );
    }
    std::env::remove_var(g4m::G4MOE_FLASH_DEPTHGATE_ENV);
    std::env::set_var(g4m::G4MOE_FLASH_DEPTHGATE_SPLITS_ENV, "47");
    assert_eq!(
        g4m::flash_depthgate_splits_for_context_tokens(
            8192,
            SHALLOW_SPLITS_8_IS_THE_ROUND_10_RECORD_ARM_PIN_TUNED_AT_256
        ),
        47,
        "the forced-splits env must override every band above the knee so one build sweeps the \
         whole ladder"
    );
    assert_eq!(
        g4m::flash_depthgate_splits_for_context_tokens(
            8,
            SHALLOW_SPLITS_8_IS_THE_ROUND_10_RECORD_ARM_PIN_TUNED_AT_256
        ),
        SHALLOW_SPLITS_8_IS_THE_ROUND_10_RECORD_ARM_PIN_TUNED_AT_256,
        "the forced-splits env must not reach below the knee"
    );
    std::env::remove_var(g4m::G4MOE_FLASH_DEPTHGATE_SPLITS_ENV);
}

fn grids_after_step_at(pos: usize, gate: bool) -> Option<(u32, u32)> {
    let _g = env_lock();
    ctx_or_skip()?;
    let mut m = tiny_model(gate, 0x2b);
    assert!(
        !m.full_attention_flash1_pass_labels().is_empty(),
        "no full-attention flash1 pass was recorded, so this suite would assert on nothing"
    );
    m.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(pos)
        .expect("synthetic fill");
    m.decode_step(1).expect("decode step");
    Some((
        m.depthgate_full_attention_flash1_grid_y_as_last_encoded()
            .expect("a full-attention flash1 pass was recorded a moment ago"),
        m.depthgate_sliding_flash1_grid_y_as_last_encoded()
            .expect("a sliding flash1 pass was recorded a moment ago"),
    ))
}

#[test]
fn gate_off_the_grid_is_the_pinned_split_count_at_every_depth_and_gate_on_it_follows_the_bands() {
    let shallow = SHALLOW_SPLITS_8_IS_THE_ROUND_10_RECORD_ARM_PIN_TUNED_AT_256;
    let Some(off_shallow) = grids_after_step_at(256, false) else {
        return;
    };
    let off_deep = grids_after_step_at(1100, false).expect("adapter was present a moment ago");
    assert_eq!(
        (off_shallow, off_deep),
        ((shallow, shallow), (shallow, shallow)),
        "with the gate off the split count is read once at build and must never move; a default \
         that moved would be an unannounced performance-default flip"
    );
    let on_shallow = grids_after_step_at(256, true).expect("adapter was present a moment ago");
    assert_eq!(
        on_shallow,
        (shallow, shallow),
        "below the knee the armed gate must dispatch the pinned shallow grid on both layer kinds"
    );
    let on_deep = grids_after_step_at(1100, true).expect("adapter was present a moment ago");
    assert_eq!(
        on_deep.0,
        g4m::FLASH_DEPTHGATE_BAND3_SPLITS_32_IS_512_WORKGROUPS_THE_MEASURED_RESIDENCY_CEILING_AND_47_AND_64_LOSE_ABOVE_IT,
        "past the third knee the armed gate must widen the full-attention stage1 grid to the \
         measured band; a grid that stayed at 8 is the 2.2x-at-122880 defect the gate exists to fix"
    );
    assert_eq!(
        on_deep.1,
        g4m::FLASH_DEPTHGATE_BAND1_SPLITS_12_IS_192_WORKGROUPS_ONE_PER_SM_AT_16_Q_HEADS_ON_188_SMS,
        "a sliding layer scans at most its window, so its band is keyed on min(total, window) = \
         640 and must stay in band 1 while the full-attention layer climbs to band 3"
    );
}

#[test]
fn arming_the_gate_below_the_knee_is_bit_for_bit_the_unarmed_decode_because_the_bands_start_above_it(
) {
    let steps = 8usize;
    let run = |gate: bool| -> Option<Vec<Vec<u32>>> {
        let _g = env_lock();
        ctx_or_skip()?;
        let mut m = tiny_model(gate, 0x51ed);
        let mut tok = 3u32;
        let mut out = Vec::new();
        for _ in 0..steps {
            let (t, logits) = m.decode_step_logits(tok).expect("decode step");
            out.push(logits.iter().map(|v| v.to_bits()).collect::<Vec<u32>>());
            tok = t;
        }
        Some(out)
    };
    let Some(off) = run(false) else { return };
    let on = run(true).expect("adapter was present a moment ago");
    assert_eq!(off.len(), steps, "the reference run produced no logits");
    assert_eq!(
        on, off,
        "the armed gate below its first knee must be the unarmed decode bit for bit; any \
         difference means the gate moved a grid it had no measured licence to move"
    );
}

const TINY_MAX_SEQ_10400_CLEARS_THE_FOLDGATE_KNEE_AT_10240_WITH_ROOM_FOR_STEPS: usize = 10400;

fn subgroup32_or_skip() -> Option<&'static nv_kernels::wgpu_backend::WgpuContext> {
    let ctx = ctx_or_skip()?;
    if !(ctx.caps.subgroup && ctx.subgroup_width() == Some(32)) {
        eprintln!("skip: foldgate needs subgroup width 32, adapter reports {:?}", ctx.subgroup_width());
        return None;
    }
    Some(ctx)
}

#[test]
fn the_foldgate_is_default_off_and_only_the_exact_string_1_arms_it() {
    let _g = env_lock();
    std::env::remove_var(g4m::G4MOE_FLASH_FOLDGATE_ENV);
    assert!(
        !g4m::flash_foldgate_enabled_full_attention_swaps_to_the_fold2_arm_at_depth_while_sliding_keeps_its_own_band(),
        "the foldgate changes which split-k arm serves full attention at depth, so it must never \
         be the default until a round flips it deliberately"
    );
    for off in ["0", "", "yes", "true", "2"] {
        std::env::set_var(g4m::G4MOE_FLASH_FOLDGATE_ENV, off);
        assert!(
            !g4m::flash_foldgate_enabled_full_attention_swaps_to_the_fold2_arm_at_depth_while_sliding_keeps_its_own_band(),
            "only the exact string 1 arms the foldgate; {off:?} must not"
        );
    }
    std::env::set_var(g4m::G4MOE_FLASH_FOLDGATE_ENV, "1");
    assert!(
        g4m::flash_foldgate_enabled_full_attention_swaps_to_the_fold2_arm_at_depth_while_sliding_keeps_its_own_band()
    );
    std::env::remove_var(g4m::G4MOE_FLASH_FOLDGATE_ENV);
}

#[test]
fn the_foldgate_knee_sits_above_every_measured_fold_arm_loss_and_below_both_standing_targets() {
    let knee = g4m::FOLDGATE_KNEE_TOKENS_10240_IS_THE_MEASURED_SIGN_CROSS_EVERY_FOLD_ARM_LOSS_SITS_SHALLOWER_AND_BOTH_STANDING_TARGETS_16K_AND_196K_SIT_DEEPER;
    assert!(
        knee > 8192 && knee <= 12288,
        "the safe window is 8192..12288: the fold arm still loses at 8192 and already wins at \
         12288, so a knee outside the window puts a measured loss in the deep band or forfeits a \
         measured win"
    );
    for measured_loss_depth in [256u32, 1024, 4096, 8192] {
        assert!(
            !g4m::foldgate_deep_at_these_whole_context_tokens(measured_loss_depth),
            "depth {measured_loss_depth} is a measured fold-arm loss and must stay on the control arm"
        );
    }
    for standing_target in [16384u32, 196608] {
        assert!(
            g4m::foldgate_deep_at_these_whole_context_tokens(standing_target),
            "depth {standing_target} is a standing target where the fold arm wins; the gate must \
             serve it the deep arm"
        );
    }
    assert!(
        !g4m::foldgate_deep_at_these_whole_context_tokens(knee - 1)
            && g4m::foldgate_deep_at_these_whole_context_tokens(knee),
        "the knee constant and the band function disagree"
    );
    let deep = g4m::FOLDGATE_DEEP_SPLITS_64_RESTORES_512_WORKGROUPS_UNDER_FOLD_2_AND_IS_A_MEASURED_RUNG_THE_LADDER_HAS_A_REPRODUCED_POTHOLE_AT_40_47_48_SO_NEVER_INTERPOLATE;
    assert!(
        ![40u32, 47, 48].contains(&deep),
        "the split ladder has a reproduced, unexplained 10-21% pothole at N = 40/47/48; the deep \
         rung must never land there"
    );
    assert_eq!(
        deep, 64,
        "the deep rung is the arm's measured 64; a different rung must be measured at that rung, \
         never interpolated from neighbours"
    );
}

#[test]
fn arming_the_foldgate_grows_only_the_full_attention_scratch_and_an_unarmed_build_owes_zero_extra_vram(
) {
    let _g = env_lock();
    std::env::set_var(g4m::G4MOE_FLASH_FOLDGATE_ENV, "1");
    assert_eq!(
        g4m::foldgate_scratch_splits_full_attention_only_must_also_cover_the_deep_arm_rows(8),
        64,
        "an armed foldgate can dispatch 64 stage1 splits on a full-attention layer, so its \
         scratch must provision 64 rows"
    );
    assert_eq!(
        g4m::foldgate_scratch_splits_full_attention_only_must_also_cover_the_deep_arm_rows(100),
        100,
        "a caller pin above the deep rung already covers it"
    );
    std::env::remove_var(g4m::G4MOE_FLASH_FOLDGATE_ENV);
    assert_eq!(
        g4m::foldgate_scratch_splits_full_attention_only_must_also_cover_the_deep_arm_rows(8),
        8,
        "with the foldgate off the scratch must not grow: an unarmed build owes zero extra VRAM"
    );
}

#[test]
fn the_foldgate_swaps_only_the_full_attention_flash1_to_fold2_splits64_at_the_knee_and_swaps_back()
{
    let _g = env_lock();
    let Some(_ctx) = subgroup32_or_skip() else {
        return;
    };
    let knee = g4m::FOLDGATE_KNEE_TOKENS_10240_IS_THE_MEASURED_SIGN_CROSS_EVERY_FOLD_ARM_LOSS_SITS_SHALLOWER_AND_BOTH_STANDING_TARGETS_16K_AND_196K_SIT_DEEPER
        as usize;
    let mut m = tiny_model_gated(
        "fgswap",
        0x77,
        TINY_MAX_SEQ_10400_CLEARS_THE_FOLDGATE_KNEE_AT_10240_WITH_ROOM_FOR_STEPS,
        None,
        Some("1"),
        None,
    );
    let shallow = SHALLOW_SPLITS_8_IS_THE_ROUND_10_RECORD_ARM_PIN_TUNED_AT_256;
    let deep = g4m::FOLDGATE_DEEP_SPLITS_64_RESTORES_512_WORKGROUPS_UNDER_FOLD_2_AND_IS_A_MEASURED_RUNG_THE_LADDER_HAS_A_REPRODUCED_POTHOLE_AT_40_47_48_SO_NEVER_INTERPOLATE;
    m.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(knee - 2)
        .expect("synthetic fill below the knee");
    m.decode_step(1).expect("decode step below the knee");
    assert!(
        !m.foldgate_deep_arm_swapped_in_as_of_last_step(),
        "one token below the knee the control arm must still be serving"
    );
    assert_eq!(
        m.foldgate_full_attention_flash1_grid_xy_as_last_encoded(),
        Some((N_Q as u32, shallow)),
        "below the knee the full-attention grid must be the caller's shallow pin on the unfolded \
         entry"
    );
    assert_eq!(
        m.depthgate_sliding_flash1_grid_y_as_last_encoded(),
        Some(shallow),
        "the sliding kind must sit on its own shallow pin below the knee"
    );
    m.decode_step(1).expect("decode step crossing the knee");
    assert!(
        m.foldgate_deep_arm_swapped_in_as_of_last_step(),
        "at the knee the deep arm must swap in"
    );
    assert_eq!(
        m.foldgate_full_attention_flash1_grid_xy_as_last_encoded(),
        Some((N_Q as u32 / 2, deep)),
        "at the knee the full-attention stage1 must dispatch the fold-2 grid at the deep split \
         rung; a grid still at the shallow pin is the starved dispatch the gate exists to fix"
    );
    assert_eq!(
        m.depthgate_sliding_flash1_grid_y_as_last_encoded(),
        Some(shallow),
        "the sliding kind must never take the fold's split count: forcing it through the shared \
         depthgate env was measured +7.2% at 1024, which is what this per-kind gate exists to avoid"
    );
    m.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(256)
        .expect("synthetic refill to a shallow depth");
    m.decode_step(1).expect("decode step after refilling shallow");
    assert!(
        !m.foldgate_deep_arm_swapped_in_as_of_last_step(),
        "a refill below the knee must swap the control arm back in"
    );
    assert_eq!(
        m.foldgate_full_attention_flash1_grid_xy_as_last_encoded(),
        Some((N_Q as u32, shallow)),
        "after swapping back the full-attention grid must be the original shallow dispatch"
    );
}

#[test]
fn arming_the_foldgate_below_the_knee_is_bit_for_bit_the_unarmed_decode() {
    let steps = 8usize;
    let run = |fg: bool| -> Option<Vec<Vec<u32>>> {
        let _g = env_lock();
        subgroup32_or_skip()?;
        let mut m = tiny_model_gated(
            if fg { "fgbit-on" } else { "fgbit-off" },
            0x51ed,
            TINY_MAX_SEQ_1200_CLEARS_KNEE3_AT_1024_WITH_ROOM_FOR_TIMED_STEPS,
            None,
            fg.then_some("1"),
            None,
        );
        let mut tok = 3u32;
        let mut out = Vec::new();
        for _ in 0..steps {
            let (t, logits) = m.decode_step_logits(tok).expect("decode step");
            out.push(logits.iter().map(|v| v.to_bits()).collect::<Vec<u32>>());
            tok = t;
        }
        Some(out)
    };
    let Some(off) = run(false) else { return };
    let on = run(true).expect("adapter was present a moment ago");
    assert_eq!(off.len(), steps, "the reference run produced no logits");
    assert_eq!(
        on, off,
        "the armed foldgate below its knee must be the unarmed decode bit for bit; any difference \
         means the deep arm leaked into the control band"
    );
}

#[test]
fn above_the_knee_the_foldgate_owes_a_reassociation_check_not_a_bitwise_one() {
    let _g = env_lock();
    let knee = g4m::FOLDGATE_KNEE_TOKENS_10240_IS_THE_MEASURED_SIGN_CROSS_EVERY_FOLD_ARM_LOSS_SITS_SHALLOWER_AND_BOTH_STANDING_TARGETS_16K_AND_196K_SIT_DEEPER
        as usize;
    let logits_for = |fg: bool| -> Option<Vec<f32>> {
        subgroup32_or_skip()?;
        let mut m = tiny_model_gated(
            if fg { "fgre-on" } else { "fgre-off" },
            0x2b,
            TINY_MAX_SEQ_10400_CLEARS_THE_FOLDGATE_KNEE_AT_10240_WITH_ROOM_FOR_STEPS,
            None,
            fg.then_some("1"),
            None,
        );
        m.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(knee + 10)
            .expect("synthetic fill above the knee");
        let (_t, logits) = m.decode_step_logits(3).expect("decode step above the knee");
        Some(logits)
    };
    let Some(ctl) = logits_for(false) else { return };
    let fold = logits_for(true).expect("adapter was present a moment ago");
    assert_eq!(ctl.len(), fold.len(), "the two arms scored different vocabs");
    let mut max_abs = 0f32;
    for (a, b) in ctl.iter().zip(fold.iter()) {
        assert!(
            a.is_finite() && b.is_finite(),
            "an arm produced a non-finite logit above the knee"
        );
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs <= 0.05,
        "fold 2 / sacc 128 / splits 64 against the unfolded shallow pin reassociates the split-k \
         sum, which moves digits, not answers; a gap of {max_abs} is a wrong band, a scratch \
         overrun, or a stale uniform, not float noise"
    );
}

#[test]
fn a_preencoded_command_buffer_from_one_band_is_discarded_at_the_knee_never_replayed_under_the_new_band(
) {
    let _g = env_lock();
    let Some(_ctx) = ctx_or_skip() else {
        return;
    };
    let knee1 = g4m::FLASH_DEPTHGATE_KNEE_TOKENS_512_IS_WHERE_12_FIRST_BEATS_THE_SHALLOW_PIN_AND_BELOW_IT_8_WINS
        as usize;
    let mut m = tiny_model_gated(
        "preenc-knee",
        0x2b,
        TINY_MAX_SEQ_1200_CLEARS_KNEE3_AT_1024_WITH_ROOM_FOR_TIMED_STEPS,
        Some("1"),
        None,
        Some("1"),
    );
    m.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(knee1 - 3)
        .expect("synthetic fill below the first knee");
    m.decode_step(1).expect("first step primes the cache");
    assert_eq!(
        m.preenc_replay_and_band_discard_counts(),
        (0, 0),
        "no command buffer existed before the first step, so nothing can replay or discard"
    );
    m.decode_step(1).expect("second step still below the knee");
    assert_eq!(
        m.preenc_replay_and_band_discard_counts(),
        (1, 0),
        "the second step shares the first step's band, so the cached command buffer must replay"
    );
    m.decode_step(1).expect("third step crosses the knee");
    assert_eq!(
        m.preenc_replay_and_band_discard_counts(),
        (1, 1),
        "the third step crosses the 512 knee: a command buffer whose stage1 grids were encoded \
         under the shallow band must be discarded, because replaying it under the widened uniform \
         makes stage2 scan scratch rows stage1 never wrote"
    );
    m.decode_step(1).expect("fourth step inside the new band");
    assert_eq!(
        m.preenc_replay_and_band_discard_counts(),
        (2, 1),
        "once re-encoded under the new band the cache must resume replaying"
    );
}
