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
const VOCAB: usize = 160;
const N_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const MOE_INTER: usize = 64;
const WINDOW: usize = 8;
const TINY_MAX_SEQ: usize = 384;
const TINY_PREFILL_M_PINNED: usize = 16;

const DECODE_STEPS_160_SO_SPLIT0_RUNS_TWO_ROUNDS_OF_THE_16X8_GRID: usize = 160;

const LOGIT_ABS_TOL_1E2_REASSOCIATED_SPLIT_SUMS_AND_EXP2_BASIS_FORBID_BITWISE_EQUALITY: f32 = 1e-2;

fn tiny_config_json() -> String {
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
        ("NV_G4MOE_FLASH_DECODE", flash.then_some("1")),
        ("NV_G4MOE_FLASH_SPLITS", splits),
        ("NV_G4MOE_KIND_LABELS", kind_labels.then_some("1")),
        ("NV_G4MOE_KV_FP8", Some(if kv_fp8 { "1" } else { "0" })),
        ("NV_G4MOE_KV_RING", None),
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

#[test]
fn flash_wgsl_parses_validates_and_exposes_both_entries_without_a_gpu() {
    let src = nv_kernels::wgpu_backend::compose(FLASH_WGSL);
    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("g4m_flash_decode.wgsl parse: {}", e.message()));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("g4m_flash_decode.wgsl validate: {e}"));
    let names: Vec<&str> = module.entry_points.iter().map(|e| e.name.as_str()).collect();
    for entry in [
        "g4m_flash_stage1_bf16",
        "g4m_flash_stage1_fp8",
        "g4m_flash_stage2_pk",
    ] {
        assert!(
            names.contains(&entry),
            "entry {entry} vanished from g4m_flash_decode.wgsl; the recorded decode passes \
             dispatch it by name (have: {names:?})"
        );
    }
}

#[test]
fn flash_gate_defaults_off_and_replaces_the_serial_arm_on_full_attention_layers_only() {
    let _g = env_lock();
    assert!(
        !G4MOE_FLASH_DECODE_DEFAULT_ON,
        "the flash decode arm must stay opt-in: g4moe-wgpu numbers on record were measured on \
         the serial g4m_attn_decode arm"
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
fn flash_decode_splits_env_matches_the_serial_arm_and_the_default_split_count() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    assert_eq!(
        nv_models::gemma4_moe_wgpu::flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build(),
        nv_models::gemma4_moe_wgpu::G4MOE_FLASH_SPLITS_16_MIRRORS_GEMMA4_WGPU_FLASH_SPLITS,
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
