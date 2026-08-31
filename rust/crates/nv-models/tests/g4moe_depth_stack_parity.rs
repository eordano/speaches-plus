#![cfg(feature = "wgpu")]

mod common;
use candle_core::Device;
use common::ctx_or_skip_no_require as ctx_or_skip;
use common::tensors_for_cfg_two_sided as tensors_for_cfg;
use common::EnvPins;
use common::LcgShift33TwoSided as Lcg;
use common::TempDir;
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::{
    sliding_kv_ring_rows_window_plus_prefill_chunk_plus_headroom, Gemma4MoeWgpu,
};
use nv_weights::WeightLoader;

const HIDDEN: usize = 64;
const INTER: usize = 96;
const N_Q: usize = 4;
const N_KV: usize = 2;
const N_GLOBAL_KV: usize = 1;
const HEAD_DIM: usize = 16;
const GLOBAL_HEAD_DIM: usize = 32;
const VOCAB: usize = 160;
const N_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const MOE_INTER: usize = 64;
const WINDOW_8_SO_THE_RING_WRAPS_WITHIN_A_FAST_TEST: usize = 8;

const TINY_MAX_SEQ_PAST_THE_152_SLOT_RING: usize = 384;
const TINY_PREFILL_M_PINNED: usize = 16;

const LOGIT_ABS_TOL_1E2_SPLIT_SUMS_AND_EXP2_FORBID_BITWISE_WHEN_FLASH_JOINS_THE_STACK: f32 = 1e-2;

fn tiny_config_json() -> String {
    format!(
        r#"{{
  "model_type": "gemma4",
  "tie_word_embeddings": true,
  "text_config": {{
    "attention_k_eq_v": true,
    "enable_moe_block": true,
    "final_logit_softcapping": 30.0,
    "global_head_dim": {GLOBAL_HEAD_DIM},
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
    "num_hidden_layers": 3,
    "num_key_value_heads": {N_KV},
    "rms_norm_eps": 1e-06,
    "rope_parameters": {{
      "full_attention": {{"partial_rotary_factor": 0.25, "rope_theta": 1000000.0}},
      "sliding_attention": {{"rope_theta": 10000.0}}
    }},
    "sliding_window": {WINDOW_8_SO_THE_RING_WRAPS_WITHIN_A_FAST_TEST},
    "tie_word_embeddings": true,
    "top_k_experts": {TOP_K},
    "vocab_size": {VOCAB}
  }}
}}"#
    )
}

fn temp_dir(tag: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("g4m_stack_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn tiny_model_stack(depth_stack_on: bool, seed: u64) -> Gemma4MoeWgpu {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let dir = temp_dir(if depth_stack_on { "on" } else { "off" });
    let st = dir.0.join("model.safetensors");
    candle_core::safetensors::save(&tensors_for_cfg(&cfg, seed), &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
    let m_s = TINY_PREFILL_M_PINNED.to_string();
    let arm = if depth_stack_on { Some("1") } else { Some("0") };
    let pins = EnvPins::pin(&[
        ("NV_G4MOE_FLASH_DECODE", arm),
        ("NV_G4MOE_FLASH_SLIDING", arm),
        ("NV_G4MOE_KV_RING", arm),
        ("NV_G4MOE_KV_FP8", Some("1")),
        ("NV_G4MOE_FLASH_SPLITS", None),
        ("NV_G4MOE_KIND_LABELS", None),
        ("NV_G4MOE_GROUPED_GEMV", None),
        ("NV_G4MOE_WGPU_PREFILL_M", Some(m_s.as_str())),
        ("NV_WGPU_PREFILL_M", None),
        ("NV_G4MOE_GEMV_WIDE", None),
        ("NV_G4MOE_W4_GEMV", None),
        ("NV_G4MOE_ROUTER_TOPK", None),
        ("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED", None),
    ]);
    let m = Gemma4MoeWgpu::from_loader(cfg, &loader, TINY_MAX_SEQ_PAST_THE_152_SLOT_RING);
    drop(pins);
    m.unwrap()
}

fn tiny_ring_rows() -> usize {
    sliding_kv_ring_rows_window_plus_prefill_chunk_plus_headroom(
        WINDOW_8_SO_THE_RING_WRAPS_WITHIN_A_FAST_TEST,
        TINY_PREFILL_M_PINNED,
    )
}

fn label_count(m: &Gemma4MoeWgpu, label: &str) -> usize {
    m.pass_rows().iter().filter(|(l, _, _, _)| l == label).count()
}

fn assert_flash_engaged_on_every_layer_so_the_walk_is_not_vacuously_serial_vs_serial(
    m: &Gemma4MoeWgpu,
    stage1_entry_fragment: &str,
) {
    let flash1 = m
        .pass_rows()
        .iter()
        .filter(|(l, e, _, _)| l.starts_with("g4m-at-flash1") && e.contains(stage1_entry_fragment))
        .count();
    assert_eq!(
        flash1, 3,
        "all 3 layers (2 sliding + 1 full) must route through flash stage1 ({stage1_entry_fragment}); \
         a serial-vs-serial walk would pass on identical bits and prove nothing"
    );
    assert_eq!(
        label_count(m, "g4m-at-decode"),
        0,
        "no layer may keep the serial decode pass once the flash arms are pinned on"
    );
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

fn assert_stack_tracks_serial(off: &[(u32, Vec<f32>)], on: &[(u32, Vec<f32>)], what: &str) {
    assert_eq!(off.len(), on.len());
    let mut worst = 0f32;
    for (i, ((tok_off, l_off), (tok_on, l_on))) in off.iter().zip(on.iter()).enumerate() {
        assert_eq!(
            tok_off, tok_on,
            "{what}: argmax diverged at step {i}; both arms read the same fp8-quantized cache \
             values, so the stack must select the same token as the serial flat arm"
        );
        for (a, b) in l_off.iter().zip(l_on.iter()) {
            worst = worst.max((a - b).abs());
        }
        assert!(
            worst <= LOGIT_ABS_TOL_1E2_SPLIT_SUMS_AND_EXP2_FORBID_BITWISE_WHEN_FLASH_JOINS_THE_STACK,
            "{what}: step {i} worst logit diff {worst}; ring slots are bit-identical and the \
             cache holds identical fp8 bytes, so a diff past split-sum reassociation means the \
             stack read the wrong slots"
        );
    }
    eprintln!("{what}: worst logit abs diff {worst:.3e} over {} steps", off.len());
}

const REAL_HEAD_DIM_256: usize = 256;
const REAL_GLOBAL_HEAD_DIM_512_FILLS_GF_MAX_HD_EXACTLY: usize = 512;
const REAL_N_Q_16: usize = 16;
const REAL_N_KV_8: usize = 8;
const REAL_N_GLOBAL_KV_2: usize = 2;

fn real_head_geometry_config_json() -> String {
    format!(
        r#"{{
  "model_type": "gemma4",
  "tie_word_embeddings": true,
  "text_config": {{
    "attention_k_eq_v": true,
    "enable_moe_block": true,
    "final_logit_softcapping": 30.0,
    "global_head_dim": {REAL_GLOBAL_HEAD_DIM_512_FILLS_GF_MAX_HD_EXACTLY},
    "head_dim": {REAL_HEAD_DIM_256},
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": {HIDDEN},
    "intermediate_size": {INTER},
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
    "max_position_embeddings": 1024,
    "moe_intermediate_size": {MOE_INTER},
    "num_attention_heads": {REAL_N_Q_16},
    "num_experts": {N_EXPERTS},
    "num_global_key_value_heads": {REAL_N_GLOBAL_KV_2},
    "num_hidden_layers": 3,
    "num_key_value_heads": {REAL_N_KV_8},
    "rms_norm_eps": 1e-06,
    "rope_parameters": {{
      "full_attention": {{"partial_rotary_factor": 0.25, "rope_theta": 1000000.0}},
      "sliding_attention": {{"rope_theta": 10000.0}}
    }},
    "sliding_window": {WINDOW_8_SO_THE_RING_WRAPS_WITHIN_A_FAST_TEST},
    "tie_word_embeddings": true,
    "top_k_experts": {TOP_K},
    "vocab_size": {VOCAB}
  }}
}}"#
    )
}

fn real_head_geometry_model(flash_on: bool, seed: u64) -> Gemma4MoeWgpu {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&real_head_geometry_config_json()).unwrap();
    let dir = temp_dir(if flash_on { "rg_on" } else { "rg_off" });
    let st = dir.0.join("model.safetensors");
    candle_core::safetensors::save(&tensors_for_cfg(&cfg, seed), &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
    let m_s = TINY_PREFILL_M_PINNED.to_string();
    let arm = if flash_on { Some("1") } else { Some("0") };
    let pins = EnvPins::pin(&[
        ("NV_G4MOE_FLASH_DECODE", arm),
        ("NV_G4MOE_FLASH_SLIDING", arm),
        ("NV_G4MOE_KV_RING", Some("0")),
        ("NV_G4MOE_KV_FP8", Some("0")),
        ("NV_G4MOE_FLASH_SPLITS", None),
        ("NV_G4MOE_KIND_LABELS", None),
        ("NV_G4MOE_GROUPED_GEMV", None),
        ("NV_G4MOE_WGPU_PREFILL_M", Some(m_s.as_str())),
        ("NV_WGPU_PREFILL_M", None),
        ("NV_G4MOE_GEMV_WIDE", None),
        ("NV_G4MOE_W4_GEMV", None),
        ("NV_G4MOE_ROUTER_TOPK", None),
        ("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED", None),
    ]);
    let m = Gemma4MoeWgpu::from_loader(cfg, &loader, TINY_MAX_SEQ_PAST_THE_152_SLOT_RING);
    drop(pins);
    m.unwrap()
}

#[test]
fn flash_matches_the_serial_arm_on_the_real_head_geometry_hd256_global512_group8() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = 160;
    let mut serial = real_head_geometry_model(false, 0x2b);
    let mut flash = real_head_geometry_model(true, 0x2b);
    assert_flash_engaged_on_every_layer_so_the_walk_is_not_vacuously_serial_vs_serial(
        &flash,
        "g4m_flash_stage1_bf16",
    );
    let bits_serial = forced_token_walk(&mut serial, steps, 0x517);
    let bits_flash = forced_token_walk(&mut flash, steps, 0x517);
    assert_stack_tracks_serial(
        &bits_serial,
        &bits_flash,
        "real-head-geometry decode walk (flash+sliding-flash bf16 flat vs serial): the tiny \
         gates stop at head_dim 32, so only this walk exercises the 8- and 16-strip lane loops \
         the 26B checkpoint runs",
    );
}

#[test]
fn depth_stack_flash_sliding_fp8_ring_matches_the_serial_fp8_flat_arm_past_the_ring_wraparound() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = 340;
    assert!(
        steps > tiny_ring_rows() + 8,
        "the walk must run past the ring wrap or the ring leg of the stack goes untested"
    );
    let mut serial = tiny_model_stack(false, 0x2b);
    let mut stack = tiny_model_stack(true, 0x2b);
    assert_flash_engaged_on_every_layer_so_the_walk_is_not_vacuously_serial_vs_serial(
        &stack,
        "g4m_flash_stage1_fp8",
    );
    let bits_serial = forced_token_walk(&mut serial, steps, 0x517);
    let bits_stack = forced_token_walk(&mut stack, steps, 0x517);
    assert_stack_tracks_serial(
        &bits_serial,
        &bits_stack,
        "depth-stack decode walk (flash+sliding-flash+fp8+ring vs serial fp8 flat)",
    );
}

#[test]
fn depth_stack_chunked_prefill_then_decode_matches_the_serial_fp8_flat_arm() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let mut rng = Lcg(0xfeed);
    let prompt: Vec<u32> = (0..320).map(|_| rng.next_u32() % VOCAB as u32).collect();
    assert!(
        prompt.len() > tiny_ring_rows(),
        "the prompt must push prefill past the ring wrap or the prefill quantize path \
         never exercises wrapped slots"
    );
    let mut serial = tiny_model_stack(false, 0x2b);
    let mut stack = tiny_model_stack(true, 0x2b);
    assert_flash_engaged_on_every_layer_so_the_walk_is_not_vacuously_serial_vs_serial(
        &stack,
        "g4m_flash_stage1_fp8",
    );
    assert!(
        serial.prefill_chunk_len() == TINY_PREFILL_M_PINNED
            && stack.prefill_chunk_len() == TINY_PREFILL_M_PINNED,
        "both arms need live chunked prefill for this gate to mean anything (got {} / {})",
        serial.prefill_chunk_len(),
        stack.prefill_chunk_len()
    );
    let done_serial = serial.prefill_tokens(&prompt).unwrap();
    let done_stack = stack.prefill_tokens(&prompt).unwrap();
    assert_eq!(done_serial, done_stack);
    assert_eq!(done_serial, prompt.len());
    let bits_serial = forced_token_walk(&mut serial, 8, 0x99);
    let bits_stack = forced_token_walk(&mut stack, 8, 0x99);
    assert_stack_tracks_serial(
        &bits_serial,
        &bits_stack,
        "depth-stack post-prefill decode (flash+sliding-flash+fp8+ring vs serial fp8 flat)",
    );
}
