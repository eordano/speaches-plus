#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip_no_require as ctx_or_skip;
use candle_core::{Device, Tensor};
use nv_models::gemma4::LayerType;
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::{
    sliding_kv_ring_rows_window_plus_prefill_chunk_plus_headroom, Gemma4MoeWgpu,
    G4MOE_KV_FP8_DEFAULT_ON, G4MOE_SLIDING_KV_RING_DEFAULT_ON,
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
const GLOBAL_HEAD_DIM: usize = 32;
const VOCAB: usize = 160;
const N_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const MOE_INTER: usize = 64;
const WINDOW_8_SO_THE_RING_WRAPS_WITHIN_A_FAST_TEST: usize = 8;

const TINY_MAX_SEQ_PAST_THE_152_SLOT_RING: usize = 384;
const TINY_PREFILL_M_PINNED: usize = 16;

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
    "num_hidden_layers": {N_LAYERS},
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
    let dir = std::env::temp_dir().join(format!("g4m_ring_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn tiny_model_arms(ring_on: bool, kv_fp8_on: bool, seed: u64) -> Gemma4MoeWgpu {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let dir = temp_dir(&format!(
        "{}_{}",
        if ring_on { "on" } else { "off" },
        if kv_fp8_on { "fp8" } else { "bf16" }
    ));
    let st = dir.0.join("model.safetensors");
    candle_core::safetensors::save(&tensors_for_cfg(&cfg, seed), &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
    let m_s = TINY_PREFILL_M_PINNED.to_string();
    let pins = EnvPins::pin(&[
        ("NV_G4MOE_KV_RING", Some(if ring_on { "1" } else { "0" })),
        ("NV_G4MOE_KV_FP8", Some(if kv_fp8_on { "1" } else { "0" })),
        ("NV_G4MOE_FLASH_DECODE", None),
        ("NV_G4MOE_WGPU_PREFILL_M", Some(m_s.as_str())),
        ("NV_WGPU_PREFILL_M", None),
        ("NV_G4MOE_GEMV_WIDE", None),
        ("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED", None),
    ]);
    let m = Gemma4MoeWgpu::from_loader(cfg, &loader, TINY_MAX_SEQ_PAST_THE_152_SLOT_RING);
    drop(pins);
    m.unwrap()
}

fn tiny_model(ring_on: bool, seed: u64) -> Gemma4MoeWgpu {
    tiny_model_arms(ring_on, false, seed)
}

fn tiny_ring_rows() -> usize {
    sliding_kv_ring_rows_window_plus_prefill_chunk_plus_headroom(
        WINDOW_8_SO_THE_RING_WRAPS_WITHIN_A_FAST_TEST,
        TINY_PREFILL_M_PINNED,
    )
}

#[test]
fn ring_gate_defaults_off_and_shrinks_only_the_sliding_layers_when_on() {
    let _g = env_lock();
    assert!(
        !G4MOE_SLIDING_KV_RING_DEFAULT_ON,
        "the sliding kv ring must stay opt-in: g4moe-wgpu numbers on record were measured on full-depth caches"
    );
    if ctx_or_skip().is_none() {
        return;
    }
    let max_seq = TINY_MAX_SEQ_PAST_THE_152_SLOT_RING;
    let ring = tiny_ring_rows();
    assert!(
        max_seq > ring,
        "test geometry must engage the ring: max_seq {max_seq} <= ring {ring}"
    );

    let off = tiny_model(false, 0x2b);
    let on = tiny_model(true, 0x2b);
    assert_eq!(
        off.state_blob_count(),
        2 * N_LAYERS,
        "expected one kc + one vc state blob per layer"
    );
    assert_eq!(on.state_blob_count(), 2 * N_LAYERS);
    let kv_words_per_row = [
        N_KV * HEAD_DIM / 2,
        N_KV * HEAD_DIM / 2,
        N_GLOBAL_KV * GLOBAL_HEAD_DIM / 2,
    ];
    for li in 0..N_LAYERS {
        let full = li == N_LAYERS - 1;
        for cache in 0..2 {
            let blob = 2 * li + cache;
            assert_eq!(
                off.state_blob_words(blob).unwrap(),
                max_seq * kv_words_per_row[li],
                "ring off must keep full-depth kv at layer {li} blob {cache}"
            );
            let expect_rows = if full { max_seq } else { ring };
            assert_eq!(
                on.state_blob_words(blob).unwrap(),
                expect_rows * kv_words_per_row[li],
                "ring on must shrink sliding layers to {ring} rows and leave full-attention at {max_seq} (layer {li} blob {cache})"
            );
        }
    }
}

fn decode_bits(m: &mut Gemma4MoeWgpu, steps: usize, seed: u64) -> Vec<u32> {
    let mut rng = Lcg(seed);
    let vocab = VOCAB as u32;
    let mut bits = Vec::new();
    for _ in 0..steps {
        let t = rng.next_u32() % vocab;
        let (tok, logits) = m.decode_step_logits(t).unwrap();
        bits.push(tok);
        bits.extend(logits.iter().map(|v| v.to_bits()));
    }
    bits
}

#[test]
fn ring_decode_is_bit_identical_to_full_depth_far_past_the_wraparound() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = 340;
    assert!(
        steps > tiny_ring_rows() + 8,
        "decode must run past the ring wrap to prove anything"
    );
    let mut off = tiny_model(false, 0x2b);
    let mut on = tiny_model(true, 0x2b);
    let bits_off = decode_bits(&mut off, steps, 0x517);
    let bits_on = decode_bits(&mut on, steps, 0x517);
    let diff = bits_off
        .iter()
        .zip(bits_on.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff, 0,
        "ring decode must be bit-identical to full depth: the same values land in wrapped slots and every sliding read stays inside the window ({diff}/{} words differ)",
        bits_off.len()
    );
}

#[test]
fn ring_chunked_prefill_then_decode_is_bit_identical_to_full_depth() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let mut rng = Lcg(0xfeed);
    let prompt: Vec<u32> = (0..320).map(|_| rng.next_u32() % VOCAB as u32).collect();
    let mut off = tiny_model(false, 0x2b);
    let mut on = tiny_model(true, 0x2b);
    assert!(
        off.prefill_chunk_len() == TINY_PREFILL_M_PINNED
            && on.prefill_chunk_len() == TINY_PREFILL_M_PINNED,
        "both arms need live chunked prefill for this gate to mean anything (got {} / {})",
        off.prefill_chunk_len(),
        on.prefill_chunk_len()
    );
    assert!(
        prompt.len() > tiny_ring_rows(),
        "the prompt must push prefill past the ring wrap"
    );
    let done_off = off.prefill_tokens(&prompt).unwrap();
    let done_on = on.prefill_tokens(&prompt).unwrap();
    assert_eq!(done_off, done_on);
    assert_eq!(done_off, prompt.len());
    let bits_off = decode_bits(&mut off, 8, 0x99);
    let bits_on = decode_bits(&mut on, 8, 0x99);
    let diff = bits_off
        .iter()
        .zip(bits_on.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff, 0,
        "post-prefill ring decode must be bit-identical to full depth ({diff}/{} words differ)",
        bits_off.len()
    );
}

const ATTN_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_attn.wgsl");
const PREFILL_HEAD_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_prefill_head.wgsl");

#[test]
fn kv_fp8_attn_and_prefill_wgsl_parse_validate_and_expose_the_fp8_entries_without_a_gpu() {
    for (name, src, entries) in [
        (
            "g4m_attn.wgsl",
            ATTN_WGSL,
            ["g4m_attn_decode", "g4m_attn_decode_fp8"],
        ),
        (
            "g4m_prefill_head.wgsl",
            PREFILL_HEAD_WGSL,
            ["pm_attn", "pm_attn_fp8"],
        ),
    ] {
        let composed = nv_kernels::wgpu_backend::compose(src);
        let module = naga::front::wgsl::parse_str(&composed)
            .unwrap_or_else(|e| panic!("{name} parse: {}", e.message()));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} validate: {e}"));
        let names: Vec<&str> = module.entry_points.iter().map(|e| e.name.as_str()).collect();
        for entry in entries {
            assert!(
                names.contains(&entry),
                "entry {entry} vanished from {name}; the recorded passes dispatch it by name \
                 (have: {names:?})"
            );
        }
    }
}

#[test]
fn kv_fp8_gate_defaults_off_halves_the_cache_words_and_adds_scale_blobs() {
    let _g = env_lock();
    assert!(
        !G4MOE_KV_FP8_DEFAULT_ON,
        "fp8 kv must stay opt-in: g4moe-wgpu numbers on record were measured on bf16 caches"
    );
    if ctx_or_skip().is_none() {
        return;
    }
    let max_seq = TINY_MAX_SEQ_PAST_THE_152_SLOT_RING;
    let ring = tiny_ring_rows();
    let bf16 = tiny_model_arms(false, false, 0x2b);
    let fp8 = tiny_model_arms(false, true, 0x2b);
    let fp8_ring = tiny_model_arms(true, true, 0x2b);
    assert_eq!(bf16.state_blob_count(), 2 * N_LAYERS);
    assert_eq!(
        fp8.state_blob_count(),
        4 * N_LAYERS,
        "fp8 kv must add one k-scale and one v-scale blob per layer (kc, vc, ksc, vsc)"
    );
    let n_kv_per_layer = [N_KV, N_KV, N_GLOBAL_KV];
    let hd_per_layer = [HEAD_DIM, HEAD_DIM, GLOBAL_HEAD_DIM];
    for li in 0..N_LAYERS {
        let full = li == N_LAYERS - 1;
        let (n_kv, hd) = (n_kv_per_layer[li], hd_per_layer[li]);
        let bf16_cache_words = max_seq * n_kv * hd / 2;
        for cache in 0..2 {
            assert_eq!(
                bf16.state_blob_words(2 * li + cache).unwrap(),
                bf16_cache_words
            );
            assert_eq!(
                fp8.state_blob_words(4 * li + cache).unwrap(),
                max_seq * n_kv * hd / 4,
                "fp8 cache blob must hold one byte per element (layer {li} blob {cache})"
            );
            assert_eq!(
                fp8.state_blob_words(4 * li + 2 + cache).unwrap(),
                max_seq * n_kv,
                "fp8 scale blob must hold one f32 per (slot, kv head) row (layer {li})"
            );
            let rows = if full { max_seq } else { ring };
            assert_eq!(
                fp8_ring.state_blob_words(4 * li + cache).unwrap(),
                rows * n_kv * hd / 4,
                "ring+fp8 must shrink sliding cache blobs to {ring} rows (layer {li})"
            );
            assert_eq!(
                fp8_ring.state_blob_words(4 * li + 2 + cache).unwrap(),
                rows * n_kv,
                "ring+fp8 must shrink sliding scale blobs to {ring} rows (layer {li})"
            );
        }
    }
}

const KV_FP8_LOGIT_ABS_TOL_6_ALLOWS_E4M3_DRIFT_AND_A_ROUTER_EXPERT_FLIP_NOT_WRONG_ROW_INDEXING:
    f32 = 6.0;

#[test]
fn kv_fp8_decode_is_deterministic_finite_and_tracks_bf16_at_e4m3_tolerance() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = 48;
    let mut bf16 = tiny_model_arms(false, false, 0x2b);
    let mut fp8_a = tiny_model_arms(false, true, 0x2b);
    let mut fp8_b = tiny_model_arms(false, true, 0x2b);
    let bits_base = decode_bits(&mut bf16, steps, 0x517);
    let bits_a = decode_bits(&mut fp8_a, steps, 0x517);
    let bits_b = decode_bits(&mut fp8_b, steps, 0x517);
    assert_eq!(
        bits_a, bits_b,
        "the fp8 kv path must be deterministic run-to-run"
    );

    let words_per_step = 1 + VOCAB;
    let mut agree = 0usize;
    let mut worst_abs = 0f32;
    for step in 0..steps {
        let base = &bits_base[step * words_per_step..(step + 1) * words_per_step];
        let quant = &bits_a[step * words_per_step..(step + 1) * words_per_step];
        if base[0] == quant[0] {
            agree += 1;
        }
        for (xb, yb) in base[1..].iter().zip(quant[1..].iter()) {
            let x = f32::from_bits(*xb);
            let y = f32::from_bits(*yb);
            assert!(
                y.is_finite(),
                "step {step}: non-finite logit under fp8 kv"
            );
            worst_abs = worst_abs.max((x - y).abs());
        }
    }
    eprintln!(
        "bf16-vs-fp8 kv: argmax agreement {agree}/{steps} worst logit abs diff {worst_abs:.6e}"
    );
    assert!(
        worst_abs <= KV_FP8_LOGIT_ABS_TOL_6_ALLOWS_E4M3_DRIFT_AND_A_ROUTER_EXPERT_FLIP_NOT_WRONG_ROW_INDEXING,
        "worst logit abs diff {worst_abs} exceeds {KV_FP8_LOGIT_ABS_TOL_6_ALLOWS_E4M3_DRIFT_AND_A_ROUTER_EXPERT_FLIP_NOT_WRONG_ROW_INDEXING}: \
         e4m3 rounding plus a value-dependent top-k expert flip stays under it on this tiny \
         softcapped model, so a diff this large means the fp8 arm read the wrong rows or scales"
    );
    assert!(
        agree * 2 >= steps,
        "argmax agreement below {}/{steps}: {agree}. NOTE: this is a 3-layer model of uniform \
         random weights, so its logit margins are near zero and argmax agreement is NOT a \
         quality signal here - it only catches gross breakage. The meaningful gate is the \
         real-checkpoint chat-wrapped ppl suite (ppl_gemma4_moe)",
        steps / 2
    );
}

#[test]
fn kv_fp8_ring_decode_is_bit_identical_to_full_depth_fp8_far_past_the_wraparound() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = 340;
    assert!(
        steps > tiny_ring_rows() + 8,
        "decode must run past the ring wrap to prove anything"
    );
    let mut full = tiny_model_arms(false, true, 0x2b);
    let mut ring = tiny_model_arms(true, true, 0x2b);
    let bits_full = decode_bits(&mut full, steps, 0x517);
    let bits_ring = decode_bits(&mut ring, steps, 0x517);
    let diff = bits_full
        .iter()
        .zip(bits_ring.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff, 0,
        "ring+fp8 decode must be bit-identical to full-depth fp8: quantization is per (slot, \
         kv head) row, so the same bytes and scales land in wrapped slots ({diff}/{} words differ)",
        bits_full.len()
    );
}

#[test]
fn kv_fp8_ring_chunked_prefill_then_decode_is_bit_identical_to_full_depth_fp8() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let mut rng = Lcg(0xfeed);
    let prompt: Vec<u32> = (0..320).map(|_| rng.next_u32() % VOCAB as u32).collect();
    let mut full = tiny_model_arms(false, true, 0x2b);
    let mut ring = tiny_model_arms(true, true, 0x2b);
    assert!(
        full.prefill_chunk_len() == TINY_PREFILL_M_PINNED
            && ring.prefill_chunk_len() == TINY_PREFILL_M_PINNED,
        "both arms need live chunked prefill so pm_attn_fp8 and the prefill quantize run (got {} / {})",
        full.prefill_chunk_len(),
        ring.prefill_chunk_len()
    );
    assert!(
        prompt.len() > tiny_ring_rows(),
        "the prompt must push prefill past the ring wrap"
    );
    let done_full = full.prefill_tokens(&prompt).unwrap();
    let done_ring = ring.prefill_tokens(&prompt).unwrap();
    assert_eq!(done_full, done_ring);
    assert_eq!(done_full, prompt.len());
    let bits_full = decode_bits(&mut full, 8, 0x99);
    let bits_ring = decode_bits(&mut ring, 8, 0x99);
    let diff = bits_full
        .iter()
        .zip(bits_ring.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff, 0,
        "post-prefill ring+fp8 decode must be bit-identical to full-depth fp8 ({diff}/{} words differ)",
        bits_full.len()
    );
}
