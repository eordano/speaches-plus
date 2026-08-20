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
    Gemma4MoeWgpu, G4MOE_FLASH_SLIDING_DEFAULT_ON, G4MOE_GROUPED_GEMV_DEFAULT_ON,
};
use nv_weights::WeightLoader;

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
const WINDOW: usize = 8;
const TINY_MAX_SEQ: usize = 384;
const TINY_PREFILL_M_PINNED: usize = 16;

const DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP: usize = 96;

const FLASH_SLIDING_LOGIT_ABS_TOL_1E2_SPLIT_SUMS_AND_EXP2_FORBID_BITWISE: f32 = 1e-2;

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
    "sliding_window": {WINDOW},
    "tie_word_embeddings": true,
    "top_k_experts": {TOP_K},
    "vocab_size": {VOCAB}
  }}
}}"#
    )
}

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn temp_dir(tag: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("g4m_grouped_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

fn tiny_model_w4(
    grouped: bool,
    flash: bool,
    flash_sliding: bool,
    ring: bool,
    w4_entry: Option<&str>,
    seed: u64,
) -> Gemma4MoeWgpu {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let dir = temp_dir(&format!(
        "{}{}{}{}{}",
        u8::from(grouped),
        u8::from(flash),
        u8::from(flash_sliding),
        u8::from(ring),
        w4_entry.unwrap_or("dflt")
    ));
    let st = dir.0.join("model.safetensors");
    candle_core::safetensors::save(&tensors_for_cfg(&cfg, seed), &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
    let m_s = TINY_PREFILL_M_PINNED.to_string();
    let pins = EnvPins::pin(&[
        ("NV_G4MOE_GROUPED_GEMV", grouped.then_some("1")),
        ("NV_G4MOE_FLASH_DECODE", flash.then_some("1")),
        ("NV_G4MOE_FLASH_SLIDING", flash_sliding.then_some("1")),
        ("NV_G4MOE_FLASH_SPLITS", None),
        ("NV_G4MOE_KIND_LABELS", None),
        ("NV_G4MOE_KV_FP8", Some("0")),
        ("NV_G4MOE_KV_RING", ring.then_some("1")),
        ("NV_G4MOE_WGPU_PREFILL_M", Some(m_s.as_str())),
        ("NV_WGPU_PREFILL_M", None),
        ("NV_G4MOE_GEMV_WIDE", None),
        ("NV_G4MOE_W4_GEMV", w4_entry),
        ("NV_G4MOE_ROUTER_TOPK", None),
        ("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED", None),
    ]);
    let m = Gemma4MoeWgpu::from_loader(cfg, &loader, TINY_MAX_SEQ);
    drop(pins);
    m.unwrap()
}

fn tiny_model(grouped: bool, flash: bool, flash_sliding: bool, ring: bool, seed: u64) -> Gemma4MoeWgpu {
    tiny_model_w4(grouped, flash, flash_sliding, ring, None, seed)
}

fn label_count(m: &Gemma4MoeWgpu, label: &str) -> usize {
    m.pass_rows().iter().filter(|(l, _, _, _)| l == label).count()
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

const PROP_NORM_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_prop_norm.wgsl");
const GEMV_BF16_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_bf16.wgsl");
const GEMV_W4E_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_w4e.wgsl");
const ATTN_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_attn.wgsl");

#[test]
fn grouped_wgsl_parses_validates_and_exposes_every_fused_entry_without_a_gpu() {
    for (name, src, entries) in [
        (
            "g4m_prop_norm.wgsl",
            PROP_NORM_WGSL,
            vec![
                "g4m_norm",
                "g4m_norm_residual",
                "g4m_norm_mul",
                "g4m_norm_norm_residual",
                "g4m_norm_x2",
                "g4m_norm_add_norm_resout",
            ],
        ),
        (
            "g4m_gemv_bf16.wgsl",
            GEMV_BF16_WGSL,
            vec!["g4m_gemv_bf16", "g4m_gemv_bf16_gu_gelu"],
        ),
        (
            "g4m_gemv_w4e.wgsl",
            GEMV_W4E_WGSL,
            vec![
                "g4m_gemv_w4",
                "g4m_gemv_w4_r8",
                "g4m_gemv_w4_gu_gelu",
                "g4m_gemv_w4_r8_gu_gelu",
                "g4m_gemv_w4_r8_down_combine",
            ],
        ),
        (
            "g4m_attn.wgsl",
            ATTN_WGSL,
            vec![
                "g4m_attn_norm_rope",
                "g4m_attn_norm_rope_qkv",
                "g4m_kv_write",
                "g4m_kv_write_stacked",
            ],
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
            assert!(names.contains(&entry), "{name} lost entry {entry} (have {names:?})");
        }
    }
}

#[test]
fn grouped_and_flash_sliding_default_off_and_grouped_collapses_the_pass_list() {
    let _g = env_lock();
    assert!(
        !G4MOE_GROUPED_GEMV_DEFAULT_ON,
        "the grouped-gemv arm must stay opt-in: every g4moe-wgpu number on record was \
         measured on the unfused pass list"
    );
    assert!(
        !G4MOE_FLASH_SLIDING_DEFAULT_ON,
        "the sliding flash arm must stay opt-in: gemma4_moe_dispatch_budget prices \
         g4m-at-decode on the serial arm"
    );
    if ctx_or_skip().is_none() {
        return;
    }
    let off = tiny_model(false, false, false, false, 0x2b);
    let on = tiny_model(true, false, false, false, 0x2b);
    for gone in [
        "g4m-moe-gelu",
        "g4m-moe-rmul",
        "g4m-moe-rnorm",
        "g4m-mlp-gelu",
        "g4m-rms-postattn",
        "g4m-pnres",
        "g4m-rms-postffw1",
        "g4m-rms-preffw2",
        "g4m-rms-postffw2",
        "g4m-ffw-add",
        "g4m-rms-postffw",
        "g4m-res-out",
        "g4m-at-qproj",
        "g4m-at-kproj",
        "g4m-at-vproj",
        "g4m-at-qnorm",
        "g4m-at-knorm",
        "g4m-at-vnorm",
    ] {
        assert!(label_count(&off, gone) > 0, "baseline lost class {gone}");
        assert_eq!(label_count(&on, gone), 0, "grouped arm still records {gone}");
    }
    for (kept, per_layer) in [
        ("g4m-moe-gu", N_LAYERS),
        ("g4m-mlp-gu", N_LAYERS),
        ("g4m-moe-rnormmul", N_LAYERS),
        ("g4m-nnres", N_LAYERS),
        ("g4m-normx2", N_LAYERS),
        ("g4m-tailnorm", N_LAYERS),
        ("g4m-at-qkvproj", N_LAYERS),
        ("g4m-at-qkvnorm", N_LAYERS),
    ] {
        assert_eq!(label_count(&on, kept), per_layer, "grouped class {kept}");
    }
    let moe_down_fused = label_count(&on, "g4m-moe-downcomb");
    let moe_down_pair = label_count(&on, "g4m-moe-down");
    assert_eq!(
        moe_down_fused + moe_down_pair,
        N_LAYERS,
        "every layer routes moe-down through exactly one arm"
    );
    if moe_down_fused == N_LAYERS {
        assert_eq!(label_count(&on, "g4m-moe-combine"), 0);
    }
    assert!(
        on.pass_count() + 12 * N_LAYERS <= off.pass_count(),
        "grouped arm must retire >= 12 dispatches per layer, got {} -> {}",
        off.pass_count(),
        on.pass_count()
    );
}

#[test]
fn grouped_gemv_is_bit_identical_to_the_unfused_pass_list_over_a_decode_walk() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut off = tiny_model(false, false, false, false, 0x2b);
    let mut on = tiny_model(true, false, false, false, 0x2b);
    let walk_off = forced_token_walk(&mut off, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    for (i, ((t_off, l_off), (t_on, l_on))) in walk_off.iter().zip(walk_on.iter()).enumerate() {
        assert_eq!(t_off, t_on, "argmax diverged at step {i}");
        assert!(
            l_off
                .iter()
                .zip(l_on.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "step {i}: grouped logits are not bit-identical; every fused stage rounds \
             through bf16 at exactly the old buffer boundaries, so any diff is a wiring bug"
        );
    }
    eprintln!(
        "grouped walk: {} steps bit-identical ({} -> {} dispatches/token)",
        steps,
        off.pass_count(),
        on.pass_count()
    );
}

#[test]
fn grouped_gemv_pair_entry_the_real_checkpoint_shape_is_bit_identical_too() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut off = tiny_model_w4(false, false, false, false, Some("wide"), 0x2b);
    let mut on = tiny_model_w4(true, false, false, false, Some("wide"), 0x2b);
    assert_eq!(
        label_count(&on, "g4m-moe-downcomb"),
        0,
        "forcing the pair w4 entry must route moe-down through the unfused fallback"
    );
    assert_eq!(label_count(&on, "g4m-moe-down"), N_LAYERS);
    assert_eq!(label_count(&on, "g4m-moe-combine"), N_LAYERS);
    let walk_off = forced_token_walk(&mut off, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    for (i, ((t_off, l_off), (t_on, l_on))) in walk_off.iter().zip(walk_on.iter()).enumerate() {
        assert_eq!(t_off, t_on, "pair-entry argmax diverged at step {i}");
        assert!(
            l_off
                .iter()
                .zip(l_on.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "pair-entry step {i}: grouped logits are not bit-identical"
        );
    }
}

#[test]
fn grouped_gemv_stays_bit_identical_after_chunked_prefill() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let mut rng = Lcg(0xfeed);
    let prompt: Vec<u32> = (0..192).map(|_| rng.next_u32() % VOCAB as u32).collect();
    let mut off = tiny_model(false, false, false, false, 0x2b);
    let mut on = tiny_model(true, false, false, false, 0x2b);
    assert_eq!(off.prefill_tokens(&prompt).unwrap(), prompt.len());
    assert_eq!(on.prefill_tokens(&prompt).unwrap(), prompt.len());
    let walk_off = forced_token_walk(&mut off, 8, 0x99);
    let walk_on = forced_token_walk(&mut on, 8, 0x99);
    for (i, ((t_off, l_off), (t_on, l_on))) in walk_off.iter().zip(walk_on.iter()).enumerate() {
        assert_eq!(t_off, t_on, "post-prefill argmax diverged at step {i}");
        assert!(
            l_off
                .iter()
                .zip(l_on.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "post-prefill step {i}: grouped logits are not bit-identical"
        );
    }
}

fn assert_argmax_and_tol(
    serial: &[(u32, Vec<f32>)],
    flash: &[(u32, Vec<f32>)],
    what: &str,
) {
    let mut worst = 0f32;
    for (i, ((t_s, l_s), (t_f, l_f))) in serial.iter().zip(flash.iter()).enumerate() {
        assert_eq!(t_s, t_f, "{what}: argmax diverged at step {i}");
        for (a, b) in l_s.iter().zip(l_f.iter()) {
            worst = worst.max((a - b).abs());
        }
        assert!(
            worst <= FLASH_SLIDING_LOGIT_ABS_TOL_1E2_SPLIT_SUMS_AND_EXP2_FORBID_BITWISE,
            "{what}: step {i} worst logit diff {worst}"
        );
    }
    eprintln!("{what}: worst logit abs diff {worst:.3e} over {} steps", serial.len());
}

#[test]
fn flash_sliding_matches_the_serial_arm_across_the_window_flat_cache() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut serial = tiny_model(false, false, false, false, 0x2b);
    let mut fl = tiny_model(false, true, true, false, 0x2b);
    assert_eq!(
        label_count(&fl, "g4m-at-decode"),
        0,
        "NV_G4MOE_FLASH_SLIDING=1 must retire the serial arm on every layer"
    );
    assert_eq!(label_count(&fl, "g4m-at-flash1"), N_LAYERS);
    let walk_serial = forced_token_walk(&mut serial, steps, 0x517);
    let walk_flash = forced_token_walk(&mut fl, steps, 0x517);
    assert_argmax_and_tol(&walk_serial, &walk_flash, "sliding flash flat-cache walk");
}

#[test]
fn flash_sliding_matches_the_serial_arm_past_the_ring_wraparound() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut serial = tiny_model(false, false, false, true, 0x2b);
    let mut fl = tiny_model(false, true, true, true, 0x2b);
    let walk_serial = forced_token_walk(&mut serial, steps, 0x517);
    let walk_flash = forced_token_walk(&mut fl, steps, 0x517);
    assert_argmax_and_tol(&walk_serial, &walk_flash, "sliding flash ring walk");
}

#[test]
fn grouped_plus_flash_sliding_the_serving_candidate_matches_the_all_off_baseline() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(false, false, false, true, 0x2b);
    let mut cand = tiny_model(true, true, true, true, 0x2b);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_cand = forced_token_walk(&mut cand, steps, 0x517);
    assert_argmax_and_tol(&walk_base, &walk_cand, "grouped+flash-sliding candidate walk");
    eprintln!(
        "candidate dispatches/token: {} -> {}",
        base.pass_count(),
        cand.pass_count()
    );
}
