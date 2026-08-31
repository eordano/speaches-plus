#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip_no_require as ctx_or_skip;
use candle_core::{Device, Tensor};
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::{
    Gemma4MoeWgpu, G4MOE_FLASH_DECODE_DEFAULT_ON, G4MOE_KIND_LABELS_DEFAULT_ON,
};
use nv_weights::WeightLoader;
use std::collections::HashMap;
use common::EnvPins;
use common::LcgShift33TwoSided as Lcg;
use common::TempDir;
use common::norm_tensor_two_sided as norm_tensor;
use common::rand_tensor_two_sided as rand_tensor;
use common::tensors_for_cfg_two_sided as tensors_for_cfg;

const HIDDEN: usize = 64;
const INTER: usize = 96;
const N_LAYERS: usize = 3;
const N_Q: usize = 4;
const N_KV: usize = 2;
const N_GLOBAL_KV: usize = 1;
const HEAD_DIM: usize = 16;
const GLOBAL_HEAD_DIM_32_ONE_FULL_LANE_STRIP: usize = 32;
const GLOBAL_HEAD_DIM_512_IS_THE_REAL_26B_WIDTH_AND_THE_ONLY_ONE_THAT_SPANS_EVERY_SACC_CHUNK: usize = 512;
const VOCAB: usize = 160;
const N_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const MOE_INTER: usize = 64;
const WINDOW: usize = 8;
const TINY_MAX_SEQ: usize = 384;
const TINY_PREFILL_M_PINNED: usize = 16;

const DECODE_STEPS_160_SO_SPLIT0_RUNS_TWO_ROUNDS_OF_THE_16X8_GRID: usize = 160;

const LOGIT_ABS_TOL_1E2_REASSOCIATED_SPLIT_SUMS_AND_EXP2_BASIS_FORBID_BITWISE_EQUALITY: f32 = 1e-2;

fn tiny_config_json_hd(global_head_dim: usize) -> String {
    format!(
        r#"{{
  "model_type": "gemma4",
  "tie_word_embeddings": true,
  "text_config": {{
    "attention_k_eq_v": true,
    "enable_moe_block": true,
    "final_logit_softcapping": 30.0,
    "global_head_dim": {global_head_dim},
    "head_dim": {HEAD_DIM},
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": {HIDDEN},
    "intermediate_size": {INTER},
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
    "max_position_embeddings": 1024,
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
    "sliding_window": {WINDOW},
    "tie_word_embeddings": true,
    "top_k_experts": {TOP_K},
    "vocab_size": {VOCAB}
  }}
}}"#
    )
}

fn tiny_config_json() -> String {
    tiny_config_json_hd(GLOBAL_HEAD_DIM_32_ONE_FULL_LANE_STRIP)
}

fn temp_dir(tag: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("g4m_flash_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn tiny_model_kv_splits(
    flash: bool,
    kind_labels: bool,
    kv_fp8: bool,
    splits: Option<&str>,
    seed: u64,
) -> Gemma4MoeWgpu {
    tiny_model_kv_splits_reduce(flash, kind_labels, kv_fp8, splits, false, seed)
}

fn tiny_model_kv_splits_reduce(
    flash: bool,
    kind_labels: bool,
    kv_fp8: bool,
    splits: Option<&str>,
    flash_sg: bool,
    seed: u64,
) -> Gemma4MoeWgpu {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let dir = temp_dir(&format!(
        "{}_{}_{}",
        if flash { "on" } else { "off" },
        if kv_fp8 { "fp8" } else { "bf16" },
        splits.unwrap_or("dflt")
    ));
    let st = dir.0.join("model.safetensors");
    candle_core::safetensors::save(&tensors_for_cfg(&cfg, seed), &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
    let m_s = TINY_PREFILL_M_PINNED.to_string();
    let pins = EnvPins::pin(&[
        ("NV_G4MOE_FLASH_DECODE", Some(if flash { "1" } else { "0" })),
        ("NV_G4MOE_FLASH_SLIDING", Some("0")),
        ("NV_G4MOE_FLASH_SPLITS", splits),
        ("NV_G4MOE_FLASH_SG", Some(if flash_sg { "1" } else { "0" })),
        ("NV_G4MOE_FLASH_SACC", None),
        ("NV_G4MOE_FLASH_FOLD", None),
        ("NV_G4MOE_KIND_LABELS", kind_labels.then_some("1")),
        ("NV_G4MOE_KV_FP8", Some(if kv_fp8 { "1" } else { "0" })),
        ("NV_G4MOE_KV_RING", Some("0")),
        ("NV_G4MOE_WGPU_PREFILL_M", Some(m_s.as_str())),
        ("NV_WGPU_PREFILL_M", None),
        ("NV_G4MOE_GEMV_WIDE", None),
        ("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED", None),
    ]);
    let m = Gemma4MoeWgpu::from_loader(cfg, &loader, TINY_MAX_SEQ);
    drop(pins);
    m.unwrap()
}

fn tiny_model_sacc(
    global_head_dim: usize,
    sacc: Option<&str>,
    flash_sg: bool,
    seed: u64,
) -> Gemma4MoeWgpu {
    tiny_model_sacc_fold(global_head_dim, sacc, flash_sg, None, seed)
}

fn tiny_model_sacc_fold(
    global_head_dim: usize,
    sacc: Option<&str>,
    flash_sg: bool,
    fold: Option<&str>,
    seed: u64,
) -> Gemma4MoeWgpu {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json_hd(global_head_dim)).unwrap();
    let dir = temp_dir(&format!(
        "sacc{}_hd{global_head_dim}_{}_fold{}",
        sacc.unwrap_or("dflt"),
        if flash_sg { "sg" } else { "wb" },
        fold.unwrap_or("1")
    ));
    let st = dir.0.join("model.safetensors");
    candle_core::safetensors::save(&tensors_for_cfg(&cfg, seed), &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
    let m_s = TINY_PREFILL_M_PINNED.to_string();
    let pins = EnvPins::pin(&[
        ("NV_G4MOE_FLASH_DECODE", Some("1")),
        ("NV_G4MOE_FLASH_SLIDING", Some("0")),
        ("NV_G4MOE_FLASH_SPLITS", None),
        ("NV_G4MOE_FLASH_SG", Some(if flash_sg { "1" } else { "0" })),
        ("NV_G4MOE_FLASH_SACC", sacc),
        ("NV_G4MOE_FLASH_FOLD", fold),
        ("NV_G4MOE_KIND_LABELS", None),
        ("NV_G4MOE_KV_FP8", Some("0")),
        ("NV_G4MOE_KV_RING", Some("0")),
        ("NV_G4MOE_WGPU_PREFILL_M", Some(m_s.as_str())),
        ("NV_WGPU_PREFILL_M", None),
        ("NV_G4MOE_GEMV_WIDE", None),
        ("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED", None),
    ]);
    let m = Gemma4MoeWgpu::from_loader(cfg, &loader, TINY_MAX_SEQ);
    drop(pins);
    m.unwrap()
}

fn tiny_model_both_kinds_flashed_with_per_kind_splits(
    global_head_dim: usize,
    splits: Option<&str>,
    sliding_splits: Option<&str>,
    fold: Option<&str>,
    sliding_fold: Option<&str>,
    flash: bool,
    seed: u64,
) -> Gemma4MoeWgpu {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json_hd(global_head_dim)).unwrap();
    let dir = temp_dir(&format!(
        "perkind_hd{global_head_dim}_{}_{}_{}_{}_{}",
        if flash { "on" } else { "off" },
        splits.unwrap_or("dflt"),
        sliding_splits.unwrap_or("dflt"),
        fold.unwrap_or("1"),
        sliding_fold.unwrap_or("dflt")
    ));
    let st = dir.0.join("model.safetensors");
    candle_core::safetensors::save(&tensors_for_cfg(&cfg, seed), &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
    let m_s = TINY_PREFILL_M_PINNED.to_string();
    let pins = EnvPins::pin(&[
        ("NV_G4MOE_FLASH_DECODE", Some(if flash { "1" } else { "0" })),
        ("NV_G4MOE_FLASH_SLIDING", Some(if flash { "1" } else { "0" })),
        ("NV_G4MOE_FLASH_SPLITS", splits),
        ("NV_G4MOE_FLASH_SPLITS_SLIDING", sliding_splits),
        ("NV_G4MOE_FLASH_DEPTHGATE", Some("0")),
        ("NV_G4MOE_FLASH_SG", Some("0")),
        ("NV_G4MOE_FLASH_SACC", None),
        ("NV_G4MOE_FLASH_FOLD", fold),
        ("NV_G4MOE_FLASH_FOLD_SLIDING", sliding_fold),
        ("NV_G4MOE_KIND_LABELS", Some("1")),
        ("NV_G4MOE_KV_FP8", Some("0")),
        ("NV_G4MOE_KV_RING", Some("0")),
        ("NV_G4MOE_WGPU_PREFILL_M", Some(m_s.as_str())),
        ("NV_WGPU_PREFILL_M", None),
        ("NV_G4MOE_GEMV_WIDE", None),
        ("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED", None),
    ]);
    let m = Gemma4MoeWgpu::from_loader(cfg, &loader, TINY_MAX_SEQ);
    drop(pins);
    m.unwrap()
}

fn tiny_model_kv(flash: bool, kind_labels: bool, kv_fp8: bool, seed: u64) -> Gemma4MoeWgpu {
    tiny_model_kv_splits(flash, kind_labels, kv_fp8, None, seed)
}

fn tiny_model(flash: bool, kind_labels: bool, seed: u64) -> Gemma4MoeWgpu {
    tiny_model_kv(flash, kind_labels, false, seed)
}

fn label_count(m: &Gemma4MoeWgpu, label: &str) -> usize {
    m.pass_rows().iter().filter(|(l, _, _, _)| l == label).count()
}

const FLASH_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_flash_decode.wgsl");

const FLASH_REDUCE_WB_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_flash_reduce_wb.wgsl");

const FLASH_REDUCE_SG_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_flash_reduce_sg.wgsl");

#[test]
fn flash_wgsl_parses_validates_and_exposes_both_entries_without_a_gpu() {
    for (tag, reduce) in [("wb", FLASH_REDUCE_WB_WGSL), ("sg", FLASH_REDUCE_SG_WGSL)] {
        for chunk in [512u32, 256, 128, 64, 32] {
        let folds: String = [2u32, 4, 8]
            .iter()
            .map(|f| nv_models::gemma4_moe_wgpu::flash_stage1_fold_source(*f))
            .collect::<Vec<_>>()
            .join("\n");
        let src = nv_kernels::wgpu_backend::compose(&format!(
            "{FLASH_WGSL}\n{reduce}\n{}\n{folds}",
            nv_models::gemma4_moe_wgpu::flash_stage1_tail_source(chunk)
        ));
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("g4m_flash_decode.wgsl+{tag}+sacc{chunk} parse: {}", e.message()));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("g4m_flash_decode.wgsl+{tag}+sacc{chunk} validate: {e}"));
        let names: Vec<&str> = module.entry_points.iter().map(|e| e.name.as_str()).collect();
        for entry in [
            "g4m_flash_stage1_bf16",
            "g4m_flash_stage1_fp8",
            "g4m_flash_stage2_pk",
            "g4m_flash_stage1_bf16_fold2",
            "g4m_flash_stage1_bf16_fold4",
            "g4m_flash_stage1_bf16_fold8",
        ] {
            assert!(
                names.contains(&entry),
                "entry {entry} vanished from g4m_flash_decode.wgsl+{tag}+sacc{chunk}; the \
                 recorded decode passes dispatch it by name (have: {names:?})"
            );
        }
        }
    }
}

#[test]
fn flash_gate_defaults_on_and_replaces_the_serial_arm_on_full_attention_layers_only() {
    let _g = env_lock();
    assert!(
        G4MOE_FLASH_DECODE_DEFAULT_ON,
        "the flash decode arm ships on: the depth ladder and parity gates in perf/runs.jsonl \
         carry both arms, and NV_G4MOE_FLASH_DECODE=0 restores the serial g4m_attn_decode arm"
    );
    assert!(
        !G4MOE_KIND_LABELS_DEFAULT_ON,
        "kind-suffixed decode labels must stay opt-in: gemma4_moe_dispatch_budget prices the \
         class named exactly g4m-at-decode"
    );
    if ctx_or_skip().is_none() {
        return;
    }
    let off = tiny_model(false, false, 0x2b);
    assert_eq!(label_count(&off, "g4m-at-decode"), N_LAYERS);
    assert_eq!(label_count(&off, "g4m-at-flash1"), 0);
    assert_eq!(label_count(&off, "g4m-at-flash2"), 0);

    let on = tiny_model(true, false, 0x2b);
    assert_eq!(
        label_count(&on, "g4m-at-decode"),
        N_LAYERS - 1,
        "sliding layers must keep the serial arm under NV_G4MOE_FLASH_DECODE=1"
    );
    assert_eq!(label_count(&on, "g4m-at-flash1"), 1);
    assert_eq!(label_count(&on, "g4m-at-flash2"), 1);

    let labeled = tiny_model(false, true, 0x2b);
    assert_eq!(label_count(&labeled, "g4m-at-decode-sliding"), N_LAYERS - 1);
    assert_eq!(label_count(&labeled, "g4m-at-decode-full"), 1);
    assert_eq!(label_count(&labeled, "g4m-at-decode"), 0);
}

fn forced_token_walk(m: &mut Gemma4MoeWgpu, steps: usize, seed: u64) -> Vec<(u32, Vec<f32>)> {
    let mut rng = Lcg(seed);
    (0..steps)
        .map(|_| {
            let t = rng.next_u32() % VOCAB as u32;
            m.decode_step_logits(t).unwrap()
        })
        .collect()
}

fn assert_arms_agree(off: &[(u32, Vec<f32>)], on: &[(u32, Vec<f32>)], what: &str) {
    assert_eq!(off.len(), on.len());
    let mut worst = 0f32;
    for (i, ((tok_off, l_off), (tok_on, l_on))) in off.iter().zip(on.iter()).enumerate() {
        assert_eq!(
            tok_off, tok_on,
            "{what}: argmax diverged at step {i}; the flash split must select the same token \
             as the serial arm on every step of this walk"
        );
        for (a, b) in l_off.iter().zip(l_on.iter()) {
            worst = worst.max((a - b).abs());
        }
        assert!(
            worst <= LOGIT_ABS_TOL_1E2_REASSOCIATED_SPLIT_SUMS_AND_EXP2_BASIS_FORBID_BITWISE_EQUALITY,
            "{what}: step {i} worst logit diff {worst}; split-k reassociates the softmax sums \
             and uses exp2, so agreement is numerical, not bitwise -- a diff this large means \
             the flash arm read the wrong slots"
        );
    }
    eprintln!("{what}: worst logit abs diff {worst:.3e} over {} steps", off.len());
}

#[test]
fn flash_decode_matches_the_serial_arm_across_two_split_rounds() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_160_SO_SPLIT0_RUNS_TWO_ROUNDS_OF_THE_16X8_GRID;
    assert!(steps > 16 * 8, "the walk must run past splits*warps tokens");
    let mut off = tiny_model(false, false, 0x2b);
    let mut on = tiny_model(true, false, 0x2b);
    let bits_off = forced_token_walk(&mut off, steps, 0x517);
    let bits_on = forced_token_walk(&mut on, steps, 0x517);
    assert_arms_agree(&bits_off, &bits_on, "decode walk");
}

#[test]
fn flash_sg_reduce_is_bitwise_identical_to_the_lds_butterfly_because_the_shuffle_chain_keeps_the_same_16_8_4_2_1_operand_order(
) {
    let _g = env_lock();
    assert!(
        !nv_models::gemma4_moe_wgpu::G4MOE_FLASH_SG_DEFAULT_OFF_UNTIL_A_ROUND_PUBLISHES_ITS_LADDER,
        "the subgroup reduce must stay opt-in until a round publishes its ladder"
    );
    let Some(ctx) = ctx_or_skip() else {
        return;
    };
    if !(ctx.caps.subgroup && ctx.subgroup_width() == Some(32)) {
        eprintln!("skip: subgroup width 32 required for the sg reduce");
        return;
    }
    let steps = DECODE_STEPS_160_SO_SPLIT0_RUNS_TWO_ROUNDS_OF_THE_16X8_GRID;
    let mut wb = tiny_model_kv_splits_reduce(true, false, false, None, false, 0x2b);
    let bits_wb = forced_token_walk(&mut wb, steps, 0x517);
    drop(wb);
    let mut sg = tiny_model_kv_splits_reduce(true, false, false, None, true, 0x2b);
    let bits_sg = forced_token_walk(&mut sg, steps, 0x517);
    assert_eq!(bits_wb.len(), bits_sg.len());
    for (i, ((t_wb, l_wb), (t_sg, l_sg))) in bits_wb.iter().zip(bits_sg.iter()).enumerate() {
        assert_eq!(t_wb, t_sg, "argmax diverged at step {i}");
        let raw_wb: Vec<u32> = l_wb.iter().map(|v| v.to_bits()).collect();
        let raw_sg: Vec<u32> = l_sg.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            raw_wb, raw_sg,
            "step {i}: NV_G4MOE_FLASH_SG replaces a shared-memory butterfly whose operand order \
             is local + partner over strides 16/8/4/2/1 with subgroupShuffleXor over the same \
             strides in the same order, so every partial sum is formed from the same two floats \
             in the same order and the result must be bitwise equal, not merely close"
        );
    }
}

#[test]
fn flash_decode_splits_env_matches_the_serial_arm_and_the_default_split_count() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    assert_eq!(
        nv_models::gemma4_moe_wgpu::flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build(),
        nv_models::gemma4_moe_wgpu::G4MOE_FLASH_SPLITS_32_IS_THE_MEASURED_KNEE_AT_DEPTH,
        "with NV_G4MOE_FLASH_SPLITS unset the split count must stay 16: every flash number \
         on record was measured on the 16-split grid"
    );
    let steps = DECODE_STEPS_160_SO_SPLIT0_RUNS_TWO_ROUNDS_OF_THE_16X8_GRID;
    let mut serial = tiny_model_kv_splits(false, false, false, None, 0x2b);
    let bits_serial = forced_token_walk(&mut serial, steps, 0x517);
    drop(serial);
    for splits in ["4", "64"] {
        let mut on = tiny_model_kv_splits(true, false, false, Some(splits), 0x2b);
        let bits_on = forced_token_walk(&mut on, steps, 0x517);
        assert_arms_agree(
            &bits_serial,
            &bits_on,
            &format!("decode walk splits={splits}"),
        );
    }
}

#[test]
fn flash_decode_matches_the_serial_arm_after_chunked_prefill() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let mut rng = Lcg(0xfeed);
    let prompt: Vec<u32> = (0..192).map(|_| rng.next_u32() % VOCAB as u32).collect();
    let mut off = tiny_model(false, false, 0x2b);
    let mut on = tiny_model(true, false, 0x2b);
    assert!(
        off.prefill_chunk_len() == TINY_PREFILL_M_PINNED
            && on.prefill_chunk_len() == TINY_PREFILL_M_PINNED,
        "both arms need live chunked prefill (got {} / {})",
        off.prefill_chunk_len(),
        on.prefill_chunk_len()
    );
    let done_off = off.prefill_tokens(&prompt).unwrap();
    let done_on = on.prefill_tokens(&prompt).unwrap();
    assert_eq!(done_off, done_on);
    assert_eq!(done_off, prompt.len());
    let bits_off = forced_token_walk(&mut off, 8, 0x99);
    let bits_on = forced_token_walk(&mut on, 8, 0x99);
    assert_arms_agree(&bits_off, &bits_on, "post-prefill decode");
}

#[test]
fn flash_decode_matches_the_serial_arm_with_fp8_kv_across_two_split_rounds() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_160_SO_SPLIT0_RUNS_TWO_ROUNDS_OF_THE_16X8_GRID;
    assert!(steps > 16 * 8, "the walk must run past splits*warps tokens");
    let mut serial = tiny_model_kv(false, false, true, 0x2b);
    let mut flash = tiny_model_kv(true, false, true, 0x2b);
    assert_eq!(
        label_count(&flash, "g4m-at-flash1"),
        1,
        "the fp8 arm must still route the full-attention layer through flash stage1"
    );
    let bits_serial = forced_token_walk(&mut serial, steps, 0x517);
    let bits_flash = forced_token_walk(&mut flash, steps, 0x517);
    assert_arms_agree(
        &bits_serial,
        &bits_flash,
        "fp8 kv decode walk (both arms read the same quantized cache, so only split-sum \
         reassociation separates them)",
    );
}

#[test]
fn flash_decode_matches_the_serial_arm_with_fp8_kv_after_chunked_prefill() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let mut rng = Lcg(0xfeed);
    let prompt: Vec<u32> = (0..192).map(|_| rng.next_u32() % VOCAB as u32).collect();
    let mut serial = tiny_model_kv(false, false, true, 0x2b);
    let mut flash = tiny_model_kv(true, false, true, 0x2b);
    assert!(
        serial.prefill_chunk_len() == TINY_PREFILL_M_PINNED
            && flash.prefill_chunk_len() == TINY_PREFILL_M_PINNED,
        "both arms need live chunked prefill (got {} / {})",
        serial.prefill_chunk_len(),
        flash.prefill_chunk_len()
    );
    let done_serial = serial.prefill_tokens(&prompt).unwrap();
    let done_flash = flash.prefill_tokens(&prompt).unwrap();
    assert_eq!(done_serial, done_flash);
    assert_eq!(done_serial, prompt.len());
    let bits_serial = forced_token_walk(&mut serial, 8, 0x99);
    let bits_flash = forced_token_walk(&mut flash, 8, 0x99);
    assert_arms_agree(&bits_serial, &bits_flash, "fp8 kv post-prefill decode");
}

#[test]
fn flash_sacc_chunked_tail_is_bitwise_identical_to_the_whole_head_dim_staging_because_every_output_element_still_sums_the_same_eight_warps_in_warp_order(
) {
    let _g = env_lock();
    assert_eq!(
        nv_models::gemma4_moe_wgpu::flash_sacc_chunk_head_dim_slice_staged_per_warp_read_once_at_build(),
        nv_models::gemma4_moe_wgpu::FLASH_SACC_CHUNK_512_IS_THE_SHIPPED_WHOLE_HEAD_DIM_STAGING_AT_16KB_PER_WORKGROUP,
        "the shipped default must stage the whole head_dim in one pass: every g4moe flash number \
         on record was measured against the 16 KB gf_sacc"
    );
    let Some(ctx) = ctx_or_skip() else {
        return;
    };
    let sg_ok = ctx.caps.subgroup && ctx.subgroup_width() == Some(32);
    let hd = GLOBAL_HEAD_DIM_512_IS_THE_REAL_26B_WIDTH_AND_THE_ONLY_ONE_THAT_SPANS_EVERY_SACC_CHUNK;
    let steps = DECODE_STEPS_160_SO_SPLIT0_RUNS_TWO_ROUNDS_OF_THE_16X8_GRID;
    let mut shipped = tiny_model_sacc(hd, None, false, 0x2b);
    let bits_shipped = forced_token_walk(&mut shipped, steps, 0x517);
    drop(shipped);
    let arms: Vec<(bool, Option<&str>)> = if sg_ok {
        vec![
            (false, Some("256")),
            (false, Some("128")),
            (false, Some("64")),
            (false, Some("32")),
            (true, None),
            (true, Some("128")),
        ]
    } else {
        vec![
            (false, Some("256")),
            (false, Some("128")),
            (false, Some("64")),
            (false, Some("32")),
        ]
    };
    for (flash_sg, sacc) in arms {
        let mut arm = tiny_model_sacc(hd, sacc, flash_sg, 0x2b);
        let bits_arm = forced_token_walk(&mut arm, steps, 0x517);
        drop(arm);
        assert_eq!(bits_shipped.len(), bits_arm.len());
        let tag = format!("sacc={:?} sg={flash_sg}", sacc.unwrap_or("512"));
        for (i, ((t_s, l_s), (t_a, l_a))) in bits_shipped.iter().zip(bits_arm.iter()).enumerate() {
            assert_eq!(t_s, t_a, "{tag}: argmax diverged at step {i}");
            let raw_s: Vec<u32> = l_s.iter().map(|v| v.to_bits()).collect();
            let raw_a: Vec<u32> = l_a.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                raw_s, raw_a,
                "{tag}: step {i}: slicing gf_sacc changes only WHICH thread reduces WHICH \
                 head_dim element and WHEN the eight warp partials are staged -- each output \
                 element is still the same eight products added in warp order 0..8, so the \
                 logits must be bitwise equal, not merely close"
            );
        }
        eprintln!("{tag}: bitwise identical over {steps} steps");
    }
}

#[test]
fn flash_gqa_fold_is_bitwise_identical_to_one_workgroup_per_query_head_because_folding_reuses_each_k_and_v_element_without_reordering_any_sum(
) {
    let _g = env_lock();
    assert_eq!(
        nv_models::gemma4_moe_wgpu::flash_gqa_fold_query_heads_per_workgroup_read_once_at_build(),
        nv_models::gemma4_moe_wgpu::FLASH_GQA_FOLD_1_IS_THE_SHIPPED_ONE_WORKGROUP_PER_QUERY_HEAD_DISPATCH,
        "the shipped dispatch must stay one workgroup per query head until a round publishes \
         the fold ladder"
    );
    let Some(ctx) = ctx_or_skip() else {
        return;
    };
    if !(ctx.caps.subgroup && ctx.subgroup_width() == Some(32)) {
        eprintln!("skip: the fold arm requires subgroup width 32");
        return;
    }
    let hd = GLOBAL_HEAD_DIM_512_IS_THE_REAL_26B_WIDTH_AND_THE_ONLY_ONE_THAT_SPANS_EVERY_SACC_CHUNK;
    let steps = DECODE_STEPS_160_SO_SPLIT0_RUNS_TWO_ROUNDS_OF_THE_16X8_GRID;
    let mut unfolded = tiny_model_sacc_fold(hd, None, true, None, 0x2b);
    let bits_unfolded = forced_token_walk(&mut unfolded, steps, 0x517);
    drop(unfolded);
    for (fold, sacc) in [
        ("2", None),
        ("4", None),
        ("2", Some("128")),
        ("4", Some("128")),
    ] {
        let mut arm = tiny_model_sacc_fold(hd, sacc, true, Some(fold), 0x2b);
        let bits_arm = forced_token_walk(&mut arm, steps, 0x517);
        drop(arm);
        assert_eq!(bits_unfolded.len(), bits_arm.len());
        let tag = format!("fold={fold} sacc={}", sacc.unwrap_or("512"));
        for (i, ((t_u, l_u), (t_a, l_a))) in bits_unfolded.iter().zip(bits_arm.iter()).enumerate() {
            assert_eq!(t_u, t_a, "{tag}: argmax diverged at step {i}");
            let raw_u: Vec<u32> = l_u.iter().map(|v| v.to_bits()).collect();
            let raw_a: Vec<u32> = l_a.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                raw_u, raw_a,
                "{tag}: step {i}: folding puts several query heads of one KV group in one \
                 workgroup so each K and V element is loaded once instead of once per head, but \
                 every dot product, every online-softmax rescale and every warp partial keeps \
                 the operand order of the unfolded kernel, so the logits must be bitwise equal"
            );
        }
        eprintln!("{tag}: bitwise identical over {steps} steps");
    }
}

#[test]
fn flash_sliding_splits_env_unset_reproduces_the_single_global_split_count_for_both_kinds() {
    let _g = env_lock();
    let pins = EnvPins::pin(&[("NV_G4MOE_FLASH_SPLITS_SLIDING", None)]);
    for full in [1u32, 8, 32, 64, 128, 256] {
        assert_eq!(
            nv_models::gemma4_moe_wgpu::flash_splits_for_layer_kind_full_attention_holds_the_folded_grid_sliding_holds_the_stage2_scan(
                nv_models::gemma4::LayerType::SlidingAttention,
                full,
            ),
            full,
            "{}",
            nv_models::gemma4_moe_wgpu::FLASH_SLIDING_SPLITS_UNSET_TRACKS_THE_FULL_ATTENTION_COUNT_SO_ADDING_THIS_KNOB_CHANGES_NO_EXISTING_ARM
        );
        assert_eq!(
            nv_models::gemma4_moe_wgpu::flash_splits_for_layer_kind_full_attention_holds_the_folded_grid_sliding_holds_the_stage2_scan(
                nv_models::gemma4::LayerType::FullAttention,
                full,
            ),
            full
        );
    }
    drop(pins);
}

#[test]
fn flash_per_kind_splits_give_each_layer_kind_its_own_grid_y_and_its_own_scratch_rather_than_both_taking_the_max(
) {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let hd_full =
        GLOBAL_HEAD_DIM_512_IS_THE_REAL_26B_WIDTH_AND_THE_ONLY_ONE_THAT_SPANS_EVERY_SACC_CHUNK;
    let full_splits = 64usize;
    let sliding_splits = 32usize;
    let m = tiny_model_both_kinds_flashed_with_per_kind_splits(
        hd_full,
        Some("64"),
        Some("32"),
        None,
        None,
        true,
        0x2b,
    );

    assert_eq!(
        m.depthgate_full_attention_flash1_grid_y_as_last_encoded(),
        Some(full_splits as u32),
        "the full-attention stage1 grid.y is the split count that holds the folded grid"
    );
    assert_eq!(
        m.depthgate_sliding_flash1_grid_y_as_last_encoded(),
        Some(sliding_splits as u32),
        "the sliding stage1 grid.y must NOT inherit the full-attention split count: its scan is \
         ring-capped, so every extra split is pure g4m_flash_stage2_pk tax on the majority kind"
    );

    let (full_bytes, sliding_bytes) = m.flash_scratch_bytes_by_layer_kind_full_then_sliding();
    let want_full = (N_Q * full_splits * (hd_full + 2) * 4) as u64;
    let want_sliding = (N_Q * sliding_splits * (HEAD_DIM + 2) * 4) as u64;
    assert_eq!(
        full_bytes,
        vec![want_full; 1],
        "the tiny config has one full-attention layer and its g4m-at-fscr must be sized for its \
         own {full_splits} splits"
    );
    assert_eq!(
        sliding_bytes,
        vec![want_sliding; 2],
        "the tiny config has two sliding layers and each g4m-at-fscr must be sized for {sliding_splits} \
         splits, not for the max of the two kinds -- sizing both at the max is the defect this \
         gate exists to catch, and it is invisible to any timing measurement"
    );
    drop(m);

    let steps = DECODE_STEPS_160_SO_SPLIT0_RUNS_TWO_ROUNDS_OF_THE_16X8_GRID;
    let mut serial =
        tiny_model_both_kinds_flashed_with_per_kind_splits(
            hd_full, None, None, None, None, false, 0x2b,
        );
    let bits_serial = forced_token_walk(&mut serial, steps, 0x517);
    drop(serial);
    let mut arm = tiny_model_both_kinds_flashed_with_per_kind_splits(
        hd_full,
        Some("64"),
        Some("32"),
        None,
        None,
        true,
        0x2b,
    );
    let bits_arm = forced_token_walk(&mut arm, steps, 0x517);
    drop(arm);
    assert_arms_agree(
        &bits_serial,
        &bits_arm,
        "decode walk full_splits=64 sliding_splits=32",
    );
}

#[test]
fn flash_sliding_fold_env_unset_reproduces_the_single_global_fold_for_both_kinds() {
    let _g = env_lock();
    let pins = EnvPins::pin(&[("NV_G4MOE_FLASH_FOLD_SLIDING", None)]);
    for full in [1u32, 2, 4, 8] {
        for kind in [
            nv_models::gemma4::LayerType::FullAttention,
            nv_models::gemma4::LayerType::SlidingAttention,
        ] {
            assert_eq!(
                nv_models::gemma4_moe_wgpu::flash_gqa_fold_for_layer_kind_only_full_attention_scans_enough_rows_to_amortise_a_halved_grid(
                    kind, full,
                ),
                full,
                "{}",
                nv_models::gemma4_moe_wgpu::FLASH_SLIDING_FOLD_HALVES_A_GRID_THE_RING_IS_TOO_SHORT_TO_REFILL_AND_IS_WHAT_FORCES_THE_SLIDING_SPLIT_TAX
            );
        }
    }
    drop(pins);
}

#[test]
fn flash_per_kind_fold_leaves_the_sliding_stage1_grid_at_full_width_while_full_attention_halves_it(
) {
    let _g = env_lock();
    let Some(ctx) = ctx_or_skip() else {
        return;
    };
    if !(ctx.caps.subgroup && ctx.subgroup_width() == Some(32)) {
        eprintln!("skip: the fold arm requires subgroup width 32");
        return;
    }
    let hd_full =
        GLOBAL_HEAD_DIM_512_IS_THE_REAL_26B_WIDTH_AND_THE_ONLY_ONE_THAT_SPANS_EVERY_SACC_CHUNK;
    let both = tiny_model_both_kinds_flashed_with_per_kind_splits(
        hd_full,
        Some("64"),
        Some("32"),
        Some("2"),
        None,
        true,
        0x2b,
    );
    let rows = both.pass_rows();
    let sliding: Vec<_> = rows
        .iter()
        .filter(|(l, _, _, _)| l == "g4m-at-flash1-sliding")
        .collect();
    let full: Vec<_> = rows
        .iter()
        .filter(|(l, _, _, _)| l == "g4m-at-flash1-full")
        .collect();
    assert_eq!(sliding.len(), 2, "the tiny config has two sliding layers");
    assert_eq!(full.len(), 1, "the tiny config has one full-attention layer");
    for (_, entry, grid, _) in &sliding {
        assert_eq!(
            (entry.as_str(), grid.0),
            ("g4m_flash_stage1_bf16_fold2", (N_Q / 2) as u32),
            "a GLOBAL fold folds the sliding layers too, because their GQA group is even"
        );
    }
    drop(both);

    let split = tiny_model_both_kinds_flashed_with_per_kind_splits(
        hd_full,
        Some("64"),
        Some("32"),
        Some("2"),
        Some("1"),
        true,
        0x2b,
    );
    let rows = split.pass_rows();
    for (l, entry, grid, _) in rows.iter() {
        if l == "g4m-at-flash1-sliding" {
            assert_eq!(
                (entry.as_str(), grid.0, grid.1),
                ("g4m_flash_stage1_bf16", N_Q as u32, 32u32),
                "NV_G4MOE_FLASH_FOLD_SLIDING=1 must leave the sliding stage1 at one workgroup \
                 per query head and at its own 32 splits: the ring caps its scan, so a halved \
                 grid has no reuse to amortise it and the doubled splits it would need to \
                 restore the grid are pure g4m_flash_stage2_pk tax on the majority layer kind"
            );
        }
        if l == "g4m-at-flash1-full" {
            assert_eq!(
                (entry.as_str(), grid.0, grid.1),
                ("g4m_flash_stage1_bf16_fold2", (N_Q / 2) as u32, 64u32),
                "full attention keeps the fold and the split count that holds its grid"
            );
        }
    }
    drop(split);

    let steps = DECODE_STEPS_160_SO_SPLIT0_RUNS_TWO_ROUNDS_OF_THE_16X8_GRID;
    let mut serial = tiny_model_both_kinds_flashed_with_per_kind_splits(
        hd_full, None, None, None, None, false, 0x2b,
    );
    let bits_serial = forced_token_walk(&mut serial, steps, 0x517);
    drop(serial);
    let mut arm = tiny_model_both_kinds_flashed_with_per_kind_splits(
        hd_full,
        Some("64"),
        Some("32"),
        Some("2"),
        Some("1"),
        true,
        0x2b,
    );
    let bits_arm = forced_token_walk(&mut arm, steps, 0x517);
    drop(arm);
    assert_arms_agree(
        &bits_serial,
        &bits_arm,
        "decode walk full fold2/splits64, sliding fold1/splits32",
    );
}
