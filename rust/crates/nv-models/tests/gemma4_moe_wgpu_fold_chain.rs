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
    Gemma4MoeWgpu, G4MOE_FOLD_CHAIN_DEFAULT_ON, G4MOE_ROUTER_CHAIN_DEFAULT_ON,
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

const ROUTER_CHAIN_LOGIT_ABS_TOL_1E2_TWO_LANE_DOT_ORDER_FORBIDS_BITWISE: f32 = 1e-2;

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
    let dir = std::env::temp_dir().join(format!("g4m_foldchain_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

fn tiny_model_full(
    fold: bool,
    chain: bool,
    flash: bool,
    flash_sliding: bool,
    ring: bool,
    coal: bool,
    wide: bool,
    r8: bool,
    rpnorm: bool,
    topk_fold: bool,
    w4_entry: Option<&str>,
    seed: u64,
) -> Gemma4MoeWgpu {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let dir = temp_dir(&format!(
        "{}{}{}{}{}{}{}{}{}{}{}",
        u8::from(fold),
        u8::from(chain),
        u8::from(flash),
        u8::from(flash_sliding),
        u8::from(ring),
        u8::from(coal),
        u8::from(wide),
        u8::from(r8),
        u8::from(rpnorm),
        u8::from(topk_fold),
        w4_entry.unwrap_or("dflt")
    ));
    let st = dir.0.join("model.safetensors");
    candle_core::safetensors::save(&tensors_for_cfg(&cfg, seed), &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
    let m_s = TINY_PREFILL_M_PINNED.to_string();
    let rpfuse_ambient_or_pre_flip_off =
        std::env::var("NV_G4MOE_RPFUSE").unwrap_or_else(|_| "0".into());
    let topk_nx2_ambient_or_pre_flip_off =
        std::env::var("NV_G4MOE_TOPK_NX2").unwrap_or_else(|_| "0".into());
    let sglat_ambient_or_pre_flip_off =
        std::env::var("NV_G4MOE_GEMV_SGLAT").unwrap_or_else(|_| "0".into());
    let pins = EnvPins::pin(&[
        ("NV_G4MOE_GROUPED_GEMV", Some("1")),
        ("NV_G4MOE_RPFUSE", Some(rpfuse_ambient_or_pre_flip_off.as_str())),
        ("NV_G4MOE_TOPK_NX2", Some(topk_nx2_ambient_or_pre_flip_off.as_str())),
        ("NV_G4MOE_GEMV_SGLAT", Some(sglat_ambient_or_pre_flip_off.as_str())),
        ("NV_G4MOE_FOLD_CHAIN", fold.then_some("1")),
        ("NV_G4MOE_ROUTER_CHAIN", chain.then_some("1")),
        ("NV_G4MOE_FLASH_DECODE", flash.then_some("1")),
        ("NV_G4MOE_FLASH_SLIDING", flash_sliding.then_some("1")),
        ("NV_G4MOE_FLASH_SPLITS", None),
        ("NV_G4MOE_KIND_LABELS", None),
        ("NV_G4MOE_KV_FP8", Some("0")),
        ("NV_G4MOE_KV_RING", ring.then_some("1")),
        ("NV_G4MOE_WGPU_PREFILL_M", Some(m_s.as_str())),
        ("NV_WGPU_PREFILL_M", None),
        ("NV_G4MOE_GEMV_WIDE", wide.then_some("1")),
        ("NV_G4MOE_BF16_R8", r8.then_some("1")),
        ("NV_G4MOE_RPNORM", Some(if rpnorm { "1" } else { "0" })),
        ("NV_G4MOE_TOPK_FOLD", topk_fold.then_some("1")),
        ("NV_G4MOE_W4_GEMV", w4_entry),
        ("NV_G4MOE_W4_COAL", coal.then_some("1")),
        ("NV_G4MOE_ROUTER_TOPK", None),
        ("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED", None),
    ]);
    let m = Gemma4MoeWgpu::from_loader(cfg, &loader, TINY_MAX_SEQ);
    drop(pins);
    m.unwrap()
}

fn tiny_model(
    fold: bool,
    chain: bool,
    flash: bool,
    flash_sliding: bool,
    ring: bool,
    seed: u64,
) -> Gemma4MoeWgpu {
    tiny_model_full(fold, chain, flash, flash_sliding, ring, false, false, false, false, false, None, seed)
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

fn assert_bitwise(base: &[(u32, Vec<f32>)], cand: &[(u32, Vec<f32>)], what: &str) {
    for (i, ((t_b, l_b), (t_c, l_c))) in base.iter().zip(cand.iter()).enumerate() {
        assert_eq!(t_b, t_c, "{what}: argmax diverged at step {i}");
        assert!(
            l_b.iter()
                .zip(l_c.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{what}: step {i} logits are not bit-identical; the fold rounds through bf16 at \
             exactly the old buffer boundaries, so any diff is a wiring bug"
        );
    }
}

fn assert_argmax_and_tol(base: &[(u32, Vec<f32>)], cand: &[(u32, Vec<f32>)], what: &str) {
    let mut worst = 0f32;
    for (i, ((t_b, l_b), (t_c, l_c))) in base.iter().zip(cand.iter()).enumerate() {
        assert_eq!(t_b, t_c, "{what}: argmax diverged at step {i}");
        for (a, b) in l_b.iter().zip(l_c.iter()) {
            worst = worst.max((a - b).abs());
        }
        assert!(
            worst <= ROUTER_CHAIN_LOGIT_ABS_TOL_1E2_TWO_LANE_DOT_ORDER_FORBIDS_BITWISE,
            "{what}: step {i} worst logit diff {worst}"
        );
    }
    eprintln!(
        "{what}: worst logit abs diff {worst:.3e} over {} steps",
        base.len()
    );
}

#[test]
fn fold_chain_default_off_and_it_retires_rms_in_kvwrite_and_final_rms() {
    let _g = env_lock();
    assert!(!G4MOE_FOLD_CHAIN_DEFAULT_ON && !G4MOE_ROUTER_CHAIN_DEFAULT_ON);
    if ctx_or_skip().is_none() {
        return;
    }
    let base = tiny_model(false, false, false, false, true, 0x2b);
    let on = tiny_model(true, false, false, false, true, 0x2b);
    assert_eq!(label_count(&on, "g4m-rms-in"), 1, "only layer 0 keeps rms-in");
    assert_eq!(label_count(&on, "g4m-tailnorm-nn"), N_LAYERS);
    assert_eq!(label_count(&on, "g4m-tailnorm"), 0);
    assert_eq!(label_count(&on, "g4m-at-kvwrite"), 0);
    assert_eq!(label_count(&on, "g4m-final-rms"), 0);
    assert_eq!(
        base.pass_count() - on.pass_count(),
        2 * N_LAYERS,
        "fold retires one rms-in-or-final-rms and one kvwrite per layer"
    );
}

#[test]
fn fold_chain_is_bit_identical_to_the_grouped_pass_list_over_a_ring_wrap_walk() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(false, false, false, false, true, 0x2b);
    let mut on = tiny_model(true, false, false, false, true, 0x2b);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "fold-chain serial-attention ring walk");
}

#[test]
fn fold_chain_stays_bit_identical_under_the_flash_default_stack() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(false, false, true, true, true, 0x2b);
    let mut on = tiny_model(true, false, true, true, true, 0x2b);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "fold-chain flash+ring walk");
}

#[test]
fn fold_chain_stays_bit_identical_after_chunked_prefill() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let mut rng = Lcg(0xfeed);
    let prompt: Vec<u32> = (0..192).map(|_| rng.next_u32() % VOCAB as u32).collect();
    let mut base = tiny_model(false, false, false, false, true, 0x2b);
    let mut on = tiny_model(true, false, false, false, true, 0x2b);
    assert_eq!(base.prefill_tokens(&prompt).unwrap(), prompt.len());
    assert_eq!(on.prefill_tokens(&prompt).unwrap(), prompt.len());
    let walk_base = forced_token_walk(&mut base, 8, 0x99);
    let walk_on = forced_token_walk(&mut on, 8, 0x99);
    assert_bitwise(&walk_base, &walk_on, "fold-chain post-prefill walk");
}

#[test]
fn router_chain_collapses_three_dispatches_and_tracks_the_grouped_arm() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(false, false, false, false, true, 0x2b);
    let mut on = tiny_model(false, true, false, false, true, 0x2b);
    assert_eq!(label_count(&on, "g4m-moe-routerchain"), N_LAYERS);
    assert_eq!(label_count(&on, "g4m-moe-rnormmul"), 0);
    assert_eq!(label_count(&on, "g4m-moe-rproj"), 0);
    assert_eq!(label_count(&on, "g4m-moe-topk"), 0);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_argmax_and_tol(&walk_base, &walk_on, "router-chain ring walk");
}

#[test]
fn fold_plus_router_chain_the_candidate_stack_tracks_the_flash_default() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(false, false, true, true, true, 0x2b);
    let mut cand = tiny_model(true, true, true, true, true, 0x2b);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_cand = forced_token_walk(&mut cand, steps, 0x517);
    assert_argmax_and_tol(&walk_base, &walk_cand, "fold+router-chain flash walk");
    eprintln!(
        "candidate dispatches/token: {} -> {}",
        base.pass_count(),
        cand.pass_count()
    );
}

#[test]
fn w4_coal_down_combine_tracks_the_r8_arm() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(false, false, false, false, true, 0x2b);
    let mut on = tiny_model_full(false, false, false, false, true, true, false, false, false, false, None, 0x2b);
    assert_eq!(
        label_count(&on, "g4m-moe-downcomb"),
        N_LAYERS,
        "coal arm must still route the fused down+combine class"
    );
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_argmax_and_tol(&walk_base, &walk_on, "w4-coal downcomb walk");
}

#[test]
fn w4_coal_pair_gu_gelu_tracks_the_pair_arm() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model_full(false, false, false, false, true, false, false, false, false, false, Some("wide"), 0x2b);
    let mut on = tiny_model_full(false, false, false, false, true, true, false, false, false, false, Some("wide"), 0x2b);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_argmax_and_tol(&walk_base, &walk_on, "w4-coal pair gu walk");
}

#[test]
fn bf16_r8_gemv_tracks_the_pair_arm_under_wide_loads() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model_full(false, false, false, false, true, false, true, false, false, false, None, 0x2b);
    let mut on = tiny_model_full(false, false, false, false, true, false, true, true, false, false, None, 0x2b);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_argmax_and_tol(&walk_base, &walk_on, "bf16-r8 wide walk");
}

#[test]
fn rpnorm_fused_router_norm_is_bit_identical_to_the_grouped_arm() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(false, false, false, false, true, 0x2b);
    let mut on = tiny_model_full(false, false, false, false, true, false, false, false, true, false, None, 0x2b);
    assert_eq!(label_count(&on, "g4m-moe-rpnorm"), N_LAYERS);
    assert_eq!(label_count(&on, "g4m-moe-rnormmul"), 0);
    assert_eq!(label_count(&on, "g4m-moe-rproj"), 0);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "rpnorm ring walk");
}

#[test]
fn topk_fold_is_bit_identical_and_retires_the_topk_dispatch() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(false, false, false, false, true, 0x2b);
    let mut on = tiny_model_full(false, false, false, false, true, false, false, false, false, true, None, 0x2b);
    assert_eq!(label_count(&on, "g4m-moe-topk"), 0);
    assert_eq!(label_count(&on, "g4m-moe-gu"), N_LAYERS);
    assert_eq!(label_count(&on, "g4m-moe-downcomb"), N_LAYERS);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "topk-fold ring walk");
}

#[test]
fn full_candidate_stack_fold_rpnorm_topkfold_is_bit_identical_under_flash() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(false, false, true, true, true, 0x2b);
    let mut cand = tiny_model_full(true, false, true, true, true, false, false, false, true, true, None, 0x2b);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_cand = forced_token_walk(&mut cand, steps, 0x517);
    assert_bitwise(&walk_base, &walk_cand, "fold+rpnorm+topkfold flash walk");
    eprintln!(
        "candidate dispatches/token: {} -> {}",
        base.pass_count(),
        cand.pass_count()
    );
}

fn entry_count(m: &Gemma4MoeWgpu, entry: &str) -> usize {
    m.pass_rows().iter().filter(|(_, e, _, _)| e == entry).count()
}

fn tiny_model_with_extra_pins(
    w4_entry: Option<&'static str>,
    extra: &[(&'static str, Option<&'static str>)],
    seed: u64,
) -> Gemma4MoeWgpu {
    let pins = EnvPins::pin(extra);
    let m = tiny_model_full(
        false, false, false, false, true, false, false, false, false, false, w4_entry, seed,
    );
    drop(pins);
    m
}

#[test]
fn down_expar_default_off_engages_the_experts_parallel_entry_and_is_bit_identical() {
    let _g = env_lock();
    assert!(!nv_models::gemma4_moe_wgpu::G4MOE_DOWN_EXPAR_DEFAULT_ON);
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(false, false, false, false, true, 0x2b);
    let mut on = tiny_model_with_extra_pins(None, &[("NV_G4MOE_DOWN_EXPAR", Some("1"))], 0x2b);
    assert_eq!(
        entry_count(&base, "g4m_gemv_w4_r8_down_combine"),
        N_LAYERS,
        "base must run the serial-expert down_combine or this test compares nothing"
    );
    assert_eq!(
        entry_count(&on, "g4m_gemv_w4_down_combine_expar"),
        N_LAYERS,
        "NV_G4MOE_DOWN_EXPAR=1 must swap every down_combine to the experts-parallel entry"
    );
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "down-expar ring walk");
}

#[test]
fn gu_fusedx_default_off_engages_the_single_x_pass_entry_and_is_bit_identical() {
    let _g = env_lock();
    assert!(!nv_models::gemma4_moe_wgpu::G4MOE_GU_FUSEDX_DEFAULT_ON);
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model_with_extra_pins(Some("wide"), &[], 0x2b);
    let mut on = tiny_model_with_extra_pins(
        Some("wide"),
        &[("NV_G4MOE_GU_FUSEDX", Some("1"))],
        0x2b,
    );
    assert_eq!(
        entry_count(&base, "g4m_gemv_w4_gu_gelu"),
        N_LAYERS,
        "forced-wide base must run the two-pass gu_gelu or this test compares nothing"
    );
    assert_eq!(
        entry_count(&on, "g4m_gemv_w4_gu_gelu_fusedx"),
        N_LAYERS,
        "NV_G4MOE_GU_FUSEDX=1 must swap every gu_gelu to the single-x-pass entry"
    );
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "gu-fusedx forced-wide ring walk");
}

#[test]
fn mlpgu_fusedx_default_off_engages_the_bf16_single_x_pass_entry_and_is_bit_identical() {
    let _g = env_lock();
    assert!(
        !nv_models::gemma4_moe_wgpu::G4MOE_MLPGU_FUSEDX_MEASURED_NEGATIVE_BF16_TWO_MAT_STREAMS_DEFAULT_OFF
    );
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(false, false, false, false, true, 0x2b);
    let mut on = tiny_model_with_extra_pins(None, &[("NV_G4MOE_MLPGU_FUSEDX", Some("1"))], 0x2b);
    assert_eq!(
        entry_count(&on, "g4m_gemv_bf16_gu_gelu_fusedx"),
        N_LAYERS,
        "NV_G4MOE_MLPGU_FUSEDX=1 must swap the dense mlp gu_gelu to its single-x-pass entry"
    );
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "mlpgu-fusedx ring walk");
}

#[test]
fn bf16_acc4_default_off_engages_on_wide_gemvs_and_keeps_argmax_within_tol() {
    let _g = env_lock();
    assert!(!nv_models::gemma4_moe_wgpu::G4MOE_BF16_ACC4_MEASURED_NO_WALL_CHANGE_DEFAULT_OFF);
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model_full(false, false, false, false, true, false, true, false, false, false, None, 0x2b);
    let pins = EnvPins::pin(&[("NV_G4MOE_BF16_ACC4", Some("1"))]);
    let mut on = tiny_model_full(false, false, false, false, true, false, true, false, false, false, None, 0x2b);
    drop(pins);
    let n_acc4 = entry_count(&on, "g4m_gemv_bf16_acc4_ilp_reorders_the_wide_dot_sum");
    assert!(
        n_acc4 >= N_LAYERS,
        "NV_G4MOE_BF16_ACC4=1 under GEMV_WIDE must swap the wide bf16 gemvs, got {n_acc4}"
    );
    assert_eq!(entry_count(&base, "g4m_gemv_bf16_acc4_ilp_reorders_the_wide_dot_sum"), 0);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_argmax_and_tol(&walk_base, &walk_on, "bf16-acc4 wide walk");
}

#[test]
fn down_expar_and_gu_fusedx_stack_bitwise_on_top_of_the_record_arm_shape() {
    let _g = env_lock();
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(true, false, true, true, true, 0x2b);
    let pins = EnvPins::pin(&[
        ("NV_G4MOE_DOWN_EXPAR", Some("1")),
        ("NV_G4MOE_GU_FUSEDX", Some("1")),
    ]);
    let mut on = tiny_model(true, false, true, true, true, 0x2b);
    drop(pins);
    assert_eq!(entry_count(&on, "g4m_gemv_w4_down_combine_expar"), N_LAYERS);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "expar+fusedx on fold+flash+ring walk");
}

const ATTN_I8_STEP0_SINGLE_PASS_REL_TOL_5PCT_MATCHES_THE_GOW_INT8_BAND: f32 = 0.05;

#[test]
fn attn_i8_group32_quality_arm_engages_i8_qkv_and_oproj_and_caps_the_verify_epilogue() {
    let _g = env_lock();
    assert!(
        !nv_models::gemma4_moe_wgpu::G4MOE_ATTN_I8_DEFAULT_ON,
        "REJECT-quality on the real checkpoint: chat-wrapped 512-token ppl 17.475 acc 0.6484 \
         vs the round-8 basis 17.161 acc 0.6465 (+0.314 ppl, +1.83 pct, 1.35x the g4moe \
         ladder's own 0.232 reassociation-null spread) and greedy identity \
         lost at token 118 of 150, so the -9.1 pct decode win stays behind the env var -- the \
         gow ATTN_I8 read at-or-better ppl and was later adjudicated negative-and-removed because \
         a 36 pct ppl DROP is a distribution change this instrument cannot certify; \
         the g16 half-group selector also REJECTS: ppl 17.401 acc 0.6465 (+0.240, +1.40 pct, \
         only 1.03x the 0.232 family null so MARGINAL on ppl alone), greedy split at the same \
         token 118, wall 6.10-6.14 at 256 -- it buys back 0.074 of the g32 delta and the \
         reject stands on the greedy divergence"
    );
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(true, false, true, true, true, 0x2b);
    assert_eq!(
        entry_count(&base, "g4m_gemv_i8_v4"),
        0,
        "zero-arm build must route no int8 attn gemv or the engagement assert below is fake"
    );
    assert!(
        base.verify_max_rows() > 0,
        "base must keep the verify epilogue so the arm's cap-0 below means something"
    );
    let pins = EnvPins::pin(&[("NV_G4MOE_ATTN_I8", Some("1"))]);
    let mut on = tiny_model(true, false, true, true, true, 0x2b);
    drop(pins);
    assert_eq!(
        entry_count(&on, "g4m_gemv_i8_v4"),
        2 * N_LAYERS,
        "NV_G4MOE_ATTN_I8=1 must route the stacked qkvproj and the oproj of every layer"
    );
    for (sel, want) in [("g4m-at-qkvproj", N_LAYERS), ("g4m-at-oproj", N_LAYERS)] {
        let pins = EnvPins::pin(&[("NV_G4MOE_ATTN_I8", Some(sel))]);
        let sub = tiny_model(true, false, true, true, true, 0x2b);
        drop(pins);
        assert_eq!(
            entry_count(&sub, "g4m_gemv_i8_v4"),
            want,
            "NV_G4MOE_ATTN_I8={sel} must route only that site class, one per layer"
        );
        assert_eq!(
            sub.verify_max_rows(),
            0,
            "any int8 attention site caps the bf16 verify epilogue"
        );
    }
    assert_eq!(
        on.verify_max_rows(),
        0,
        "int8 decode vs the bf16 prefill verify epilogue cannot keep the argmax contract"
    );
    assert!(
        on.weight_bytes_per_token() < base.weight_bytes_per_token(),
        "the int8 copies must shrink the per-token weight read ledger ({} vs {})",
        on.weight_bytes_per_token(),
        base.weight_bytes_per_token()
    );
    let g16_pins = EnvPins::pin(&[("NV_G4MOE_ATTN_I8", Some("g16"))]);
    let g16 = tiny_model(true, false, true, true, true, 0x2b);
    drop(g16_pins);
    assert_eq!(
        entry_count(
            &g16,
            "g4m_gemv_i8_v4_g16_one_scale_per_vec4_word_halves_the_group_of_the_shared_i8_entry"
        ),
        2 * N_LAYERS,
        "the g16 token must route both attention sites to the half-group entry"
    );
    assert_eq!(
        nv_models::gemma4_moe_wgpu::attn_i8_group_elems(),
        nv_models::gemma4_moe_wgpu::ATTN_I8_GROUP_32_IS_THE_GOW_RECIPE_G16_HALVES_THE_QUANT_STEP_AT_DOUBLE_THE_SCALE_STREAM,
        "the group selector must fall back to 32 once the g16 pin is dropped"
    );
    assert!(
        g16.weight_bytes_per_token() > on.weight_bytes_per_token(),
        "g16 doubles the scale stream, so its per-token read must exceed g32's ({} vs {})",
        g16.weight_bytes_per_token(),
        on.weight_bytes_per_token()
    );

    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    let (l0b, l0o) = (&walk_base[0].1, &walk_on[0].1);
    let scale = l0b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst0 = l0b
        .iter()
        .zip(l0o.iter())
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst0 / scale <= ATTN_I8_STEP0_SINGLE_PASS_REL_TOL_5PCT_MATCHES_THE_GOW_INT8_BAND,
        "step-0 caches are identical, so the single-pass int8 projection error must stay \
         inside the band: worst {worst0} vs scale {scale}; later steps compound kv drift and \
         are gated by the real-checkpoint ppl suite instead"
    );
    for (i, (_t, l)) in walk_on.iter().enumerate() {
        assert!(
            l.iter().all(|v| v.is_finite()),
            "step {i}: int8-armed logits must stay finite"
        );
    }
}

fn sg32_or_skip() -> bool {
    let Some(ctx) = ctx_or_skip() else {
        return false;
    };
    if !ctx.caps.subgroup || ctx.subgroup_width() != Some(32) {
        eprintln!("skipping: norm-lat entries need subgroup width 32");
        return false;
    }
    true
}

#[test]
fn norm_lat_default_off_engages_subgroup_shuffle_norm_entries_and_is_bit_identical() {
    let _g = env_lock();
    assert!(!nv_models::gemma4_moe_wgpu::G4MOE_NORM_LAT_DEFAULT_ON);
    if !sg32_or_skip() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(true, false, true, true, true, 0x2b);
    let pins = EnvPins::pin(&[("NV_G4MOE_NORM_LAT", Some("1"))]);
    let mut on = tiny_model(true, false, true, true, true, 0x2b);
    drop(pins);
    for (entry, n) in [
        ("g4m_norm_add_norm_resout_normnext_sglat", N_LAYERS),
        ("g4m_norm_norm_residual_sglat", N_LAYERS),
        ("g4m_norm_x2_sglat", N_LAYERS),
        ("g4m_attn_norm_rope_qkv_kvwrite_sglat", N_LAYERS),
    ] {
        assert_eq!(
            entry_count(&on, entry),
            n,
            "NV_G4MOE_NORM_LAT=1 must swap every {entry} site to the subgroup-shuffle entry"
        );
        assert_eq!(
            entry_count(&base, entry),
            0,
            "base must stay on the scratch-tree entries or this test compares nothing"
        );
    }
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "norm-lat subgroup-shuffle ring walk");
}

#[test]
fn norm_lat_stacks_bitwise_with_expar_and_gu_fusedx_on_the_record_arm_shape() {
    let _g = env_lock();
    if !sg32_or_skip() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base =
        tiny_model_full(true, false, true, true, true, false, false, false, true, false, None, 0x2b);
    let pins = EnvPins::pin(&[
        ("NV_G4MOE_DOWN_EXPAR", Some("1")),
        ("NV_G4MOE_GU_FUSEDX", Some("1")),
        ("NV_G4MOE_NORM_LAT", Some("1")),
    ]);
    let mut on =
        tiny_model_full(true, false, true, true, true, false, false, false, true, false, None, 0x2b);
    drop(pins);
    assert_eq!(
        entry_count(&on, "g4m_norm_add_norm_resout_normnext_sglat"),
        N_LAYERS
    );
    assert_eq!(entry_count(&on, "g4m_gemv_bf16_rpnorm_sglat"), N_LAYERS);
    assert_eq!(entry_count(&base, "g4m_gemv_bf16_rpnorm"), N_LAYERS);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "norm-lat+expar+fusedx record-shape ring walk");
}

#[test]
fn gemv_sglat_unset_routes_at_oproj_only_and_forced_on_engages_every_subgroup_shuffle_gemv_entry_bit_identically() {
    let _g = env_lock();
    assert_eq!(
        nv_models::gemma4_moe_wgpu::G4MOE_GEMV_SGLAT_UNSET_ROUTES_AT_OPROJ_ONLY_ROW_R3_111_1_TOK_S_BROADER_LABEL_SETS_DILUTE,
        "at-oproj",
        "NV_G4MOE_GEMV_SGLAT=0 restores the pre-flip all-scratch-tree arm"
    );
    if !sg32_or_skip() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model_with_extra_pins(
        Some("wide"),
        &[("NV_G4MOE_GU_FUSEDX", Some("1"))],
        0x2b,
    );
    let mut on = tiny_model_with_extra_pins(
        Some("wide"),
        &[
            ("NV_G4MOE_GU_FUSEDX", Some("1")),
            ("NV_G4MOE_GEMV_SGLAT", Some("1")),
        ],
        0x2b,
    );
    assert_eq!(
        entry_count(&base, "g4m_gemv_w4_gu_gelu_fusedx"),
        N_LAYERS,
        "forced-wide base must run the scratch-tree fusedx or this test compares nothing"
    );
    assert_eq!(
        entry_count(&on, "g4m_gemv_w4_gu_gelu_fusedx_sglat"),
        N_LAYERS,
        "NV_G4MOE_GEMV_SGLAT=1 must swap every w4 fusedx to the subgroup-shuffle entry"
    );
    assert_eq!(
        entry_count(&on, "g4m_gemv_bf16_gu_gelu_sglat"),
        N_LAYERS,
        "NV_G4MOE_GEMV_SGLAT=1 must swap the dense mlp gu_gelu to the subgroup-shuffle entry"
    );
    assert!(
        entry_count(&on, "g4m_gemv_bf16_sglat") >= N_LAYERS,
        "NV_G4MOE_GEMV_SGLAT=1 must swap the plain bf16 gemvs to the subgroup-shuffle entry"
    );
    assert_eq!(entry_count(&base, "g4m_gemv_bf16_sglat"), 0);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "gemv-sglat forced-wide fusedx ring walk");
}

#[test]
fn gemv_sglat_swaps_the_expar_down_combine_and_is_bit_identical() {
    let _g = env_lock();
    if !sg32_or_skip() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model_with_extra_pins(None, &[("NV_G4MOE_DOWN_EXPAR", Some("1"))], 0x2b);
    let mut on = tiny_model_with_extra_pins(
        None,
        &[
            ("NV_G4MOE_DOWN_EXPAR", Some("1")),
            ("NV_G4MOE_GEMV_SGLAT", Some("1")),
        ],
        0x2b,
    );
    assert_eq!(
        entry_count(&base, "g4m_gemv_w4_down_combine_expar"),
        N_LAYERS,
        "base must run the scratch-tree expar down_combine or this test compares nothing"
    );
    assert_eq!(
        entry_count(&on, "g4m_gemv_w4_down_combine_expar_sglat"),
        N_LAYERS,
        "NV_G4MOE_GEMV_SGLAT=1 must swap every expar down_combine to the subgroup-shuffle entry"
    );
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "gemv-sglat expar ring walk");
}

#[test]
fn gemv_sglat_stacks_bitwise_on_the_record_arm_shape() {
    let _g = env_lock();
    if !sg32_or_skip() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let record_pins = [
        ("NV_G4MOE_DOWN_EXPAR", Some("1")),
        ("NV_G4MOE_GU_FUSEDX", Some("1")),
        ("NV_G4MOE_NORM_LAT", Some("1")),
    ];
    let pins = EnvPins::pin(&record_pins);
    let mut base =
        tiny_model_full(true, false, true, true, true, false, true, false, true, false, None, 0x2b);
    drop(pins);
    let mut on_pins = record_pins.to_vec();
    on_pins.push(("NV_G4MOE_GEMV_SGLAT", Some("1")));
    let pins = EnvPins::pin(&on_pins);
    let mut on =
        tiny_model_full(true, false, true, true, true, false, true, false, true, false, None, 0x2b);
    drop(pins);
    assert_eq!(
        entry_count(&on, "g4m_gemv_w4_down_combine_expar_sglat"),
        N_LAYERS
    );
    assert_eq!(entry_count(&on, "g4m_gemv_bf16_gu_gelu_sglat"), N_LAYERS);
    assert!(entry_count(&on, "g4m_gemv_bf16_sglat") >= N_LAYERS);
    assert_eq!(entry_count(&base, "g4m_gemv_bf16_sglat"), 0);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(
        &walk_base,
        &walk_on,
        "gemv-sglat stacked on fold+flash+ring+rpnorm+norm-lat record shape",
    );
}

#[test]
fn w4_xv4_default_off_swaps_the_sglat_w4_entries_to_vec4_loads_and_is_bit_identical() {
    let _g = env_lock();
    assert!(!nv_models::gemma4_moe_wgpu::G4MOE_W4_XV4_STACKS_ON_GEMV_SGLAT_DEFAULT_ON);
    if !sg32_or_skip() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model_with_extra_pins(
        Some("wide"),
        &[("NV_G4MOE_GU_FUSEDX", Some("1"))],
        0x2b,
    );
    let mut on = tiny_model_with_extra_pins(
        Some("wide"),
        &[
            ("NV_G4MOE_GU_FUSEDX", Some("1")),
            ("NV_G4MOE_GEMV_SGLAT", Some("1")),
            ("NV_G4MOE_W4_XV4", Some("1")),
        ],
        0x2b,
    );
    assert_eq!(
        entry_count(&on, "g4m_gemv_w4_gu_gelu_fusedx_sglat_xv4"),
        N_LAYERS,
        "NV_G4MOE_W4_XV4=1 on the sglat arm must swap every w4 fusedx to the vec4-load entry"
    );
    assert_eq!(entry_count(&base, "g4m_gemv_w4_gu_gelu_fusedx_sglat_xv4"), 0);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "w4-xv4 forced-wide fusedx ring walk");
}

#[test]
fn w4_xv4_swaps_the_sglat_expar_down_combine_and_is_bit_identical() {
    let _g = env_lock();
    if !sg32_or_skip() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model_with_extra_pins(None, &[("NV_G4MOE_DOWN_EXPAR", Some("1"))], 0x2b);
    let mut on = tiny_model_with_extra_pins(
        None,
        &[
            ("NV_G4MOE_DOWN_EXPAR", Some("1")),
            ("NV_G4MOE_GEMV_SGLAT", Some("1")),
            ("NV_G4MOE_W4_XV4", Some("1")),
        ],
        0x2b,
    );
    assert_eq!(
        entry_count(&on, "g4m_gemv_w4_down_combine_expar_sglat_xv4"),
        N_LAYERS,
        "NV_G4MOE_W4_XV4=1 on the sglat arm must swap every expar down_combine to the vec4-load entry"
    );
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "w4-xv4 expar ring walk");
}

#[test]
fn tailfold_default_off_folds_every_mid_stack_tailnorm_into_the_next_qkvproj_and_is_bit_identical()
{
    let _g = env_lock();
    assert!(!nv_models::gemma4_moe_wgpu::G4MOE_TAILFOLD_MEASURED_NEGATIVE_PROLOGUE_REDUNDANCY_SCALES_WITH_3072_QKVPROJ_WGS_DEFAULT_OFF);
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base =
        tiny_model_full(true, false, true, true, true, false, true, false, false, false, None, 0x2b);
    let pins = EnvPins::pin(&[("NV_G4MOE_TAILFOLD", Some("1"))]);
    let mut on =
        tiny_model_full(true, false, true, true, true, false, true, false, false, false, None, 0x2b);
    drop(pins);
    assert_eq!(
        entry_count(&on, "g4m_gemv_bf16_tailfold"),
        N_LAYERS - 1,
        "every deferred mid-stack tailnorm must ride the next layer's qkvproj"
    );
    assert_eq!(
        label_count(&on, "g4m-tailnorm-nn"),
        1,
        "only the final tailnorm (which feeds the vocab-wide lmhead) keeps its own dispatch"
    );
    assert_eq!(entry_count(&base, "g4m_gemv_bf16_tailfold"), 0);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "tailfold wide flash ring walk");
}

#[test]
fn qkvfold_default_off_folds_every_qkvnorm_into_flash_stage1_and_is_bit_identical() {
    let _g = env_lock();
    assert!(!nv_models::gemma4_moe_wgpu::G4MOE_QKVFOLD_MEASURED_NEGATIVE_SERIAL_TREE_PROLOGUE_DOUBLES_STAGE1_DEFAULT_OFF);
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(true, false, true, true, true, 0x2b);
    let pins = EnvPins::pin(&[("NV_G4MOE_QKVFOLD", Some("1"))]);
    let mut on = tiny_model(true, false, true, true, true, 0x2b);
    drop(pins);
    assert_eq!(
        entry_count(&on, "g4m_flash_stage1_bf16_qkvfold"),
        N_LAYERS,
        "NV_G4MOE_QKVFOLD=1 under full flash must swap every stage1 to the folding entry"
    );
    assert_eq!(
        label_count(&on, "g4m-at-qkvnorm"),
        0,
        "the folded stage1 must retire the standalone norm+rope+kvwrite dispatch"
    );
    assert_eq!(entry_count(&base, "g4m_flash_stage1_bf16_qkvfold"), 0);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "qkvfold flash ring walk");
}

#[test]
fn both_folds_stack_bitwise_on_the_record_arm_shape_and_after_chunked_prefill() {
    let _g = env_lock();
    if !sg32_or_skip() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let record_pins = [
        ("NV_G4MOE_DOWN_EXPAR", Some("1")),
        ("NV_G4MOE_GU_FUSEDX", Some("1")),
        ("NV_G4MOE_NORM_LAT", Some("1")),
        ("NV_G4MOE_PREENC", Some("1")),
    ];
    let pins = EnvPins::pin(&record_pins);
    let mut base =
        tiny_model_full(true, false, true, true, true, false, true, false, true, false, None, 0x2b);
    drop(pins);
    let mut on_pins = record_pins.to_vec();
    on_pins.push(("NV_G4MOE_TAILFOLD", Some("1")));
    on_pins.push(("NV_G4MOE_QKVFOLD", Some("1")));
    let pins = EnvPins::pin(&on_pins);
    let mut on =
        tiny_model_full(true, false, true, true, true, false, true, false, true, false, None, 0x2b);
    drop(pins);
    assert_eq!(entry_count(&on, "g4m_gemv_bf16_tailfold"), N_LAYERS - 1);
    assert_eq!(entry_count(&on, "g4m_flash_stage1_bf16_qkvfold"), N_LAYERS);
    let mut rng = Lcg(0xfeed);
    let prompt: Vec<u32> = (0..192).map(|_| rng.next_u32() % VOCAB as u32).collect();
    assert_eq!(base.prefill_tokens(&prompt).unwrap(), prompt.len());
    assert_eq!(on.prefill_tokens(&prompt).unwrap(), prompt.len());
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(
        &walk_base,
        &walk_on,
        "tailfold+qkvfold on the record shape, post-prefill ring walk",
    );
    eprintln!(
        "record-shape dispatches/token: {} -> {}",
        base.pass_count(),
        on.pass_count()
    );
}

#[test]
fn preenc_default_off_prebuilt_next_step_command_buffer_is_bit_identical() {
    let _g = env_lock();
    assert!(!nv_models::gemma4_moe_wgpu::G4MOE_PREENC_DEFAULT_ON);
    if ctx_or_skip().is_none() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let mut base = tiny_model(true, false, true, true, true, 0x2b);
    let pins = EnvPins::pin(&[("NV_G4MOE_PREENC", Some("1"))]);
    let mut on = tiny_model(true, false, true, true, true, 0x2b);
    drop(pins);
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "preenc pending-cb ring walk");
}

#[test]
fn rpfuse_default_on_rides_the_router_gemv_on_the_mlp_down_dispatch_and_is_bit_identical() {
    let _g = env_lock();
    assert!(
        nv_models::gemma4_moe_wgpu::G4MOE_RPFUSE_RIDES_THE_ROUTER_NORM_GEMV_ON_THE_MLP_DOWN_DISPATCH_DEFAULT_ON,
        "SHIPS ON with the fusion champion stack (rows 'g4moe fusion' r2/r3, 111.1 vs 87.3 \
         tok/s at 256); NV_G4MOE_RPFUSE=0 restores the standalone rpnorm dispatch"
    );
    if !sg32_or_skip() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let record_pins = [("NV_G4MOE_NORM_LAT", Some("1"))];
    let pins = EnvPins::pin(&record_pins);
    let mut base =
        tiny_model_full(true, false, true, true, true, false, true, false, true, false, None, 0x2b);
    drop(pins);
    let mut on_pins = record_pins.to_vec();
    on_pins.push(("NV_G4MOE_RPFUSE", Some("1")));
    let pins = EnvPins::pin(&on_pins);
    let mut on =
        tiny_model_full(true, false, true, true, true, false, true, false, true, false, None, 0x2b);
    drop(pins);
    assert_eq!(
        entry_count(&base, "g4m_gemv_bf16_rpnorm_sglat"),
        N_LAYERS,
        "base must run the standalone rpnorm dispatch or this test compares nothing"
    );
    assert_eq!(
        entry_count(&on, "g4m_gemv_bf16_down_rpnorm_sglat"),
        N_LAYERS,
        "NV_G4MOE_RPFUSE=1 must ride every router gemv on the mlp-down dispatch"
    );
    assert_eq!(entry_count(&on, "g4m_gemv_bf16_rpnorm_sglat"), 0);
    assert_eq!(label_count(&on, "g4m-mlp-down"), 0);
    assert_eq!(
        label_count(&on, "g4m-moe-topk"),
        N_LAYERS,
        "without NV_G4MOE_TOPK_NX2 the topk dispatch must survive"
    );
    assert_eq!(
        on.pass_count(),
        base.pass_count() - N_LAYERS,
        "the fusion must retire exactly one dispatch per layer"
    );
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "rpfuse down+router ring walk");
}

#[test]
fn topk_nx2_default_on_rides_the_topk_on_the_normx2_dispatch_and_is_bit_identical() {
    let _g = env_lock();
    assert!(
        nv_models::gemma4_moe_wgpu::G4MOE_TOPK_NX2_RIDES_THE_ROUTER_TOPK_ON_THE_NORMX2_DISPATCH_DEFAULT_ON,
        "SHIPS ON with the fusion champion stack (rows 'g4moe fusion' r2/r3, 111.1 vs 87.3 \
         tok/s at 256); NV_G4MOE_TOPK_NX2=0 restores the standalone topk dispatch"
    );
    if !sg32_or_skip() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let record_pins = [("NV_G4MOE_NORM_LAT", Some("1"))];
    let pins = EnvPins::pin(&record_pins);
    let mut base =
        tiny_model_full(true, false, true, true, true, false, true, false, true, false, None, 0x2b);
    drop(pins);
    let mut on_pins = record_pins.to_vec();
    on_pins.push(("NV_G4MOE_RPFUSE", Some("1")));
    on_pins.push(("NV_G4MOE_TOPK_NX2", Some("1")));
    let pins = EnvPins::pin(&on_pins);
    let mut on =
        tiny_model_full(true, false, true, true, true, false, true, false, true, false, None, 0x2b);
    drop(pins);
    assert_eq!(
        entry_count(&base, "g4m_router_topk_par"),
        N_LAYERS,
        "base must run the standalone parallel-rank topk or this test compares nothing"
    );
    assert_eq!(
        entry_count(&on, "g4m_norm_x2_topk_sglat"),
        N_LAYERS,
        "NV_G4MOE_TOPK_NX2=1 must ride every decode topk on the normx2 dispatch"
    );
    assert_eq!(entry_count(&on, "g4m_router_topk_par"), 0);
    assert_eq!(label_count(&on, "g4m-moe-topk"), 0);
    assert_eq!(label_count(&on, "g4m-normx2"), 0);
    assert_eq!(
        on.pass_count(),
        base.pass_count() - 2 * N_LAYERS,
        "the two fusions must retire exactly two dispatches per layer"
    );
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(&walk_base, &walk_on, "topk-on-normx2 ring walk");
}

#[test]
fn rpfuse_and_topk_nx2_stack_bitwise_on_the_record_arm_shape_and_after_chunked_prefill() {
    let _g = env_lock();
    if !sg32_or_skip() {
        return;
    }
    let steps = DECODE_STEPS_96_CROSS_THE_WINDOW_AND_RING_WRAP;
    let record_pins = [
        ("NV_G4MOE_DOWN_EXPAR", Some("1")),
        ("NV_G4MOE_GU_FUSEDX", Some("1")),
        ("NV_G4MOE_NORM_LAT", Some("1")),
        ("NV_G4MOE_GEMV_SGLAT", Some("1")),
        ("NV_G4MOE_PREENC", Some("1")),
    ];
    let pins = EnvPins::pin(&record_pins);
    let mut base =
        tiny_model_full(true, false, true, true, true, false, true, false, true, false, None, 0x2b);
    drop(pins);
    let mut on_pins = record_pins.to_vec();
    on_pins.push(("NV_G4MOE_RPFUSE", Some("1")));
    on_pins.push(("NV_G4MOE_TOPK_NX2", Some("1")));
    let pins = EnvPins::pin(&on_pins);
    let mut on =
        tiny_model_full(true, false, true, true, true, false, true, false, true, false, None, 0x2b);
    drop(pins);
    assert_eq!(entry_count(&on, "g4m_gemv_bf16_down_rpnorm_sglat"), N_LAYERS);
    assert_eq!(entry_count(&on, "g4m_norm_x2_topk_sglat"), N_LAYERS);
    assert_eq!(on.pass_count(), base.pass_count() - 2 * N_LAYERS);
    let mut rng = Lcg(0xfeed);
    let prompt: Vec<u32> = (0..192).map(|_| rng.next_u32() % VOCAB as u32).collect();
    assert_eq!(base.prefill_tokens(&prompt).unwrap(), prompt.len());
    assert_eq!(on.prefill_tokens(&prompt).unwrap(), prompt.len());
    let walk_base = forced_token_walk(&mut base, steps, 0x517);
    let walk_on = forced_token_walk(&mut on, steps, 0x517);
    assert_bitwise(
        &walk_base,
        &walk_on,
        "rpfuse+topk-nx2 on the record shape, post-prefill ring walk",
    );
    eprintln!(
        "record-shape dispatches/token: {} -> {}",
        base.pass_count(),
        on.pass_count()
    );
}
