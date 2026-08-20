#![cfg(feature = "wgpu")]

mod common;
use common::LcgTop24TwoSided as Lcg;
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::{dequantize_w4_expert, host_layer_from_loader, W4_GS};
use nv_quant::nvfp4::{dequantize_packed_linear, Nvfp4Tensor};
use nv_weights::WeightLoader;
use std::io::Write;
use std::path::{Path, PathBuf};

const HIDDEN: usize = 64;
const INTER: usize = 64;
const MOE_INTER: usize = 64;
const N_Q: usize = 4;
const N_KV: usize = 2;
const HEAD_DIM: usize = 16;
const N_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const VOCAB: usize = 64;

const STORED_GLOBAL_SCALE_2_SO_A_DROPPED_DIVISION_DOUBLES_EVERY_WEIGHT: f32 = 2.0;

fn tiny_config_json() -> String {
    format!(
        r#"{{
  "model_type": "gemma4",
  "tie_word_embeddings": true,
  "text_config": {{
    "attention_k_eq_v": true,
    "enable_moe_block": true,
    "final_logit_softcapping": 30.0,
    "global_head_dim": 32,
    "head_dim": {HEAD_DIM},
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": {HIDDEN},
    "intermediate_size": {INTER},
    "layer_types": ["sliding_attention"],
    "max_position_embeddings": 64,
    "moe_intermediate_size": {MOE_INTER},
    "num_attention_heads": {N_Q},
    "num_experts": {N_EXPERTS},
    "num_global_key_value_heads": 1,
    "num_hidden_layers": 1,
    "num_key_value_heads": {N_KV},
    "rms_norm_eps": 1e-06,
    "rope_parameters": {{
      "full_attention": {{"partial_rotary_factor": 0.25, "rope_theta": 1000000.0}},
      "sliding_attention": {{"rope_theta": 10000.0}}
    }},
    "sliding_window": 16,
    "tie_word_embeddings": true,
    "top_k_experts": {TOP_K},
    "vocab_size": {VOCAB}
  }}
}}"#
    )
}

fn rand_rows(rng: &mut Lcg, n: usize, k: usize, scale: f32) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..k).map(|_| rng.next_f32() * scale).collect())
        .collect()
}

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect()
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

type StEntry = (String, &'static str, Vec<usize>, Vec<u8>);

fn write_safetensors_by_hand_because_candle_save_has_no_f8e4m3(path: &Path, entries: &[StEntry]) {
    let mut header = String::from("{");
    let mut off = 0usize;
    for (i, (name, dt, shape, bytes)) in entries.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let end = off + bytes.len();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dt}\",\"shape\":{shape:?},\"data_offsets\":[{off},{end}]}}"
        ));
        off = end;
    }
    header.push('}');
    let mut hb = header.into_bytes();
    while hb.len() % 8 != 0 {
        hb.push(b' ');
    }
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&hb).unwrap();
    for (_, _, _, bytes) in entries {
        f.write_all(bytes).unwrap();
    }
}

fn push_norm(entries: &mut Vec<StEntry>, rng: &mut Lcg, name: String, dim: usize) {
    let vals: Vec<f32> = (0..dim).map(|_| 1.0 + 0.25 * rng.next_f32()).collect();
    entries.push((name, "BF16", vec![dim], bf16_bytes(&vals)));
}

fn push_ct_nvfp4_module(
    entries: &mut Vec<StEntry>,
    rng: &mut Lcg,
    module: &str,
    n: usize,
    k: usize,
) -> Nvfp4Tensor {
    let rows = rand_rows(rng, n, k, 0.5);
    let t = Nvfp4Tensor::quantize_rows_with_global(
        &rows,
        STORED_GLOBAL_SCALE_2_SO_A_DROPPED_DIVISION_DOUBLES_EVERY_WEIGHT,
    );
    entries.push((
        format!("{module}.weight_packed"),
        "U8",
        vec![n, k / 2],
        t.data.clone(),
    ));
    entries.push((
        format!("{module}.weight_scale"),
        "F8_E4M3",
        vec![n, k / 16],
        t.scales.clone(),
    ));
    entries.push((
        format!("{module}.weight_global_scale"),
        "F32",
        vec![1],
        f32_bytes(&[STORED_GLOBAL_SCALE_2_SO_A_DROPPED_DIVISION_DOUBLES_EVERY_WEIGHT]),
    ));
    entries.push((
        format!("{module}.input_global_scale"),
        "F32",
        vec![1],
        f32_bytes(&[1.0]),
    ));
    t
}

fn reference_dequant(t: &Nvfp4Tensor, n: usize, k: usize) -> Vec<f32> {
    dequantize_packed_linear(
        &t.data,
        &t.scales,
        n,
        k,
        1.0 / STORED_GLOBAL_SCALE_2_SO_A_DROPPED_DIVISION_DOUBLES_EVERY_WEIGHT,
    )
}

struct TinyCtCheckpoint {
    dir: PathBuf,
    q: Nvfp4Tensor,
    experts_gate: Vec<Nvfp4Tensor>,
    experts_up: Vec<Nvfp4Tensor>,
    experts_down: Vec<Nvfp4Tensor>,
}

fn build_tiny_ct_checkpoint(seed: u64, tag: &str, drop_tensor: Option<&str>) -> TinyCtCheckpoint {
    let p = "model.language_model.layers.0";
    let mut rng = Lcg(seed);
    let mut entries: Vec<StEntry> = Vec::new();
    for norm in [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "post_feedforward_layernorm_1",
        "pre_feedforward_layernorm_2",
        "post_feedforward_layernorm_2",
    ] {
        push_norm(&mut entries, &mut rng, format!("{p}.{norm}.weight"), HIDDEN);
    }
    entries.push((
        format!("{p}.layer_scalar"),
        "F32",
        vec![1],
        f32_bytes(&[1.0]),
    ));
    push_norm(
        &mut entries,
        &mut rng,
        format!("{p}.self_attn.q_norm.weight"),
        HEAD_DIM,
    );
    push_norm(
        &mut entries,
        &mut rng,
        format!("{p}.self_attn.k_norm.weight"),
        HEAD_DIM,
    );
    let router_vals: Vec<f32> = (0..N_EXPERTS * HIDDEN).map(|_| rng.next_f32()).collect();
    entries.push((
        format!("{p}.router.proj.weight"),
        "BF16",
        vec![N_EXPERTS, HIDDEN],
        bf16_bytes(&router_vals),
    ));
    push_norm(&mut entries, &mut rng, format!("{p}.router.scale"), HIDDEN);
    entries.push((
        format!("{p}.router.per_expert_scale"),
        "F32",
        vec![N_EXPERTS],
        f32_bytes(&vec![1.0f32; N_EXPERTS]),
    ));

    let q = push_ct_nvfp4_module(
        &mut entries,
        &mut rng,
        &format!("{p}.self_attn.q_proj"),
        N_Q * HEAD_DIM,
        HIDDEN,
    );
    for (m, n, k) in [
        (format!("{p}.self_attn.k_proj"), N_KV * HEAD_DIM, HIDDEN),
        (format!("{p}.self_attn.v_proj"), N_KV * HEAD_DIM, HIDDEN),
        (format!("{p}.self_attn.o_proj"), HIDDEN, N_Q * HEAD_DIM),
        (format!("{p}.mlp.gate_proj"), INTER, HIDDEN),
        (format!("{p}.mlp.up_proj"), INTER, HIDDEN),
        (format!("{p}.mlp.down_proj"), HIDDEN, INTER),
    ] {
        push_ct_nvfp4_module(&mut entries, &mut rng, &m, n, k);
    }
    let mut experts_gate = Vec::new();
    let mut experts_up = Vec::new();
    let mut experts_down = Vec::new();
    for e in 0..N_EXPERTS {
        experts_gate.push(push_ct_nvfp4_module(
            &mut entries,
            &mut rng,
            &format!("{p}.experts.{e}.gate_proj"),
            MOE_INTER,
            HIDDEN,
        ));
        experts_up.push(push_ct_nvfp4_module(
            &mut entries,
            &mut rng,
            &format!("{p}.experts.{e}.up_proj"),
            MOE_INTER,
            HIDDEN,
        ));
        experts_down.push(push_ct_nvfp4_module(
            &mut entries,
            &mut rng,
            &format!("{p}.experts.{e}.down_proj"),
            HIDDEN,
            MOE_INTER,
        ));
    }
    if let Some(name) = drop_tensor {
        let before = entries.len();
        entries.retain(|(n, _, _, _)| n != name);
        assert_eq!(before, entries.len() + 1, "drop_tensor {name} not present");
    }
    let dir = std::env::temp_dir().join(format!(
        "g4moe-ct-nvfp4-{}-{tag}-{seed:x}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    write_safetensors_by_hand_because_candle_save_has_no_f8e4m3(
        &dir.join("model.safetensors"),
        &entries,
    );
    TinyCtCheckpoint {
        dir,
        q,
        experts_gate,
        experts_up,
        experts_down,
    }
}

fn assert_close_to_reference_within_bf16_rounding(got_bits: &[u16], reference: &[f32], what: &str) {
    assert_eq!(got_bits.len(), reference.len(), "{what}: length mismatch");
    for (i, (g, r)) in got_bits.iter().zip(reference.iter()).enumerate() {
        let gv = half::bf16::from_bits(*g).to_f32();
        let tol = r.abs() * 0.008 + 1e-6;
        assert!(
            (gv - r).abs() <= tol,
            "{what}[{i}]: loaded {gv} vs nvfp4 reference {r} exceeds bf16 rounding tol {tol} \
             (a factor-2 gap here means weight_global_scale was not divided out)"
        );
    }
}

fn assert_expert_stack_matches_reference_within_w4_requant_step(
    got: &[f32],
    reference: &[f32],
    n: usize,
    k: usize,
    what: &str,
) {
    assert_eq!(got.len(), n * k, "{what}: dequant length");
    for r in 0..n {
        for g in 0..k / W4_GS {
            let base = r * k + g * W4_GS;
            let amax = reference[base..base + W4_GS]
                .iter()
                .fold(0f32, |a, v| a.max(v.abs()));
            let half_step_plus_bf16_slack = (amax / 7.0) * 0.6 + 1e-6;
            for j in 0..W4_GS {
                let d = (got[base + j] - reference[base + j]).abs();
                assert!(
                    d <= half_step_plus_bf16_slack,
                    "{what} row {r} group {g} elem {j}: |{} - {}| = {d} exceeds the int4 \
                     group-requant half step {half_step_plus_bf16_slack}",
                    got[base + j],
                    reference[base + j]
                );
            }
        }
    }
}

#[test]
fn tiny_unfused_ct_nvfp4_checkpoint_loads_and_matches_reference_dequant() {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let ck = build_tiny_ct_checkpoint(0xc7_4ef4_0001, "ok", None);
    let loader = WeightLoader::open_file(ck.dir.join("model.safetensors"), &candle_core::Device::Cpu)
        .unwrap();
    let layer = host_layer_from_loader(&cfg, &loader, 0).unwrap();

    let q_ref = reference_dequant(&ck.q, N_Q * HEAD_DIM, HIDDEN);
    assert_close_to_reference_within_bf16_rounding(&layer.q.w, &q_ref, "q_proj");
    assert!(layer.v.is_some(), "sliding layer must load v_proj");

    assert_eq!(layer.experts_gate.e, N_EXPERTS);
    assert_eq!(layer.experts_gate.n, MOE_INTER);
    assert_eq!(layer.experts_gate.k, HIDDEN);
    assert_eq!(layer.experts_down.n, HIDDEN);
    assert_eq!(layer.experts_down.k, MOE_INTER);
    for e in 0..N_EXPERTS {
        for (stack, refs, n, k, what) in [
            (&layer.experts_gate, &ck.experts_gate, MOE_INTER, HIDDEN, "gate"),
            (&layer.experts_up, &ck.experts_up, MOE_INTER, HIDDEN, "up"),
            (&layer.experts_down, &ck.experts_down, HIDDEN, MOE_INTER, "down"),
        ] {
            let got = dequantize_w4_expert(stack, e);
            let reference = reference_dequant(&refs[e], n, k);
            assert_expert_stack_matches_reference_within_w4_requant_step(
                &got,
                &reference,
                n,
                k,
                &format!("expert {e} {what}"),
            );
        }
    }
}

#[test]
fn tiny_unfused_ct_nvfp4_missing_scale_fails_loudly_naming_the_tensor() {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let missing = "model.language_model.layers.0.experts.1.up_proj.weight_scale";
    let ck = build_tiny_ct_checkpoint(0xc7_4ef4_0002, "missing", Some(missing));
    let loader = WeightLoader::open_file(ck.dir.join("model.safetensors"), &candle_core::Device::Cpu)
        .unwrap();
    let err = match host_layer_from_loader(&cfg, &loader, 0) {
        Ok(_) => panic!("expected a missing-tensor error"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        err.contains("experts.1.up_proj.weight_scale"),
        "error must name the missing tensor, got: {err}"
    );
}

const REDHAT_NVFP4_HOME_HUB_REPO: &str = "models--RedHatAI--gemma-4-26B-A4B-it-NVFP4";

fn redhat_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_G4MOE_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(&home)
        .join(".cache/huggingface/hub")
        .join(REDHAT_NVFP4_HOME_HUB_REPO)
        .join("snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("redhat nvfp4 snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file() && p.join("model.safetensors").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no complete RedHatAI nvfp4 snapshot under HOME hub; set NV_G4MOE_SNAPSHOT")
}

#[test]
#[ignore = "mmaps the real 16.4 GB RedHatAI nvfp4 repack; set NV_G4MOE_NVFP4_TEST=1 (and optionally NV_G4MOE_SNAPSHOT)"]
fn redhat_nvfp4_repack_layer0_loads_into_w4_expert_storage() {
    assert_eq!(
        std::env::var("NV_G4MOE_NVFP4_TEST").ok().as_deref(),
        Some("1"),
        "this gated suite must never silently skip: set NV_G4MOE_NVFP4_TEST=1"
    );
    let dir = redhat_snapshot_dir_env_override_then_home_hub();
    let cfg = Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    let loader = WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
    let layer = host_layer_from_loader(&cfg, &loader, 0).expect("layer 0 via ct-nvfp4 arm");

    assert_eq!(layer.experts_gate.e, cfg.num_experts);
    assert_eq!(layer.experts_gate.n, cfg.moe_intermediate_size);
    assert_eq!(layer.experts_gate.k, cfg.base.hidden_size);
    assert_eq!(layer.experts_down.n, cfg.base.hidden_size);
    assert_eq!(layer.experts_down.k, cfg.moe_intermediate_size);
    assert_eq!(layer.q.n, cfg.base.num_attention_heads * cfg.base.head_dim_for(cfg.base.layer_kind(0)));

    let vals = dequantize_w4_expert(&layer.experts_gate, 0);
    let finite = vals.iter().all(|v| v.is_finite());
    let nonzero = vals.iter().filter(|v| **v != 0.0).count();
    let amax = vals.iter().fold(0f32, |a, v| a.max(v.abs()));
    assert!(finite, "expert 0 gate dequant produced non-finite values");
    assert!(
        nonzero * 2 > vals.len(),
        "expert 0 gate dequant is mostly zero ({nonzero}/{})",
        vals.len()
    );
    assert!(
        amax > 1e-4 && amax < 1e3,
        "expert 0 gate dequant amax {amax} is implausible for trained weights"
    );
    eprintln!(
        "machine: g4moe-ct-nvfp4-layer0 snapshot={} experts={} mi={} hidden={} amax={amax:.4} nonzero_frac={:.3}",
        dir.display(),
        layer.experts_gate.e,
        layer.experts_gate.n,
        layer.experts_gate.k,
        nonzero as f64 / vals.len() as f64
    );
}
