#![cfg(feature = "wgpu")]

mod common;
use common::bf16_bits;
use common::bf16_val;
use common::have_gpu;
use common::HIDDEN_64 as HIDDEN;
use common::INTER_96 as INTER;
use common::LcgTop24TwoSided as Lcg;
use common::MOE_INTER;
use common::N_EXPERTS;
use common::N_Q_4 as N_Q;
use common::norm_tensor;
use common::rand_tensor_f32_shape as rand_tensor;
use common::tiny_config_json;
use common::VOCAB_160 as VOCAB;
use candle_core::{Device, Tensor};
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::{
    dequantize_i8_row, lmhead_i8_entry_off_until_real_checkpoint_quality_gate, quantize_i8_host,
    Gemma4MoeWgpu, I8_GS,
};
use nv_weights::WeightLoader;
use std::collections::HashMap;
use common::TempDir;

const ROW_CHUNK: usize = 64;
const STEPS: usize = 6;
const BF16_ULP_REL: f32 = 1.0 / 256.0;

fn q_over_128_grid_embed(rng: &mut Lcg, vocab: usize, hidden: usize) -> Vec<f32> {
    let groups = hidden / I8_GS;
    let mut w = vec![0f32; vocab * hidden];
    for r in 0..vocab {
        for g in 0..groups {
            let base = r * hidden + g * I8_GS;
            for j in 0..I8_GS {
                let q = (rng.next_f32() * 127.0).round().clamp(-127.0, 127.0);
                w[base + j] = q / 128.0;
            }
            let sign = if rng.next_f32() < 0.0 { -1.0 } else { 1.0 };
            w[base] = sign * 127.0 / 128.0;
        }
    }
    w
}

fn tiny_tensors_with_lossless_head(seed: u64) -> HashMap<String, Tensor> {
    use nv_models::gemma4::LayerType;
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let base = &cfg.base;
    let mut rng = Lcg(seed);
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let embed = q_over_128_grid_embed(&mut rng, VOCAB, HIDDEN);
    t.insert(
        "model.language_model.embed_tokens.weight".into(),
        Tensor::from_vec(embed, (VOCAB, HIDDEN), &Device::Cpu).unwrap(),
    );
    t.insert(
        "model.language_model.norm.weight".into(),
        norm_tensor(&mut rng, HIDDEN),
    );
    for i in 0..base.num_hidden_layers {
        let p = format!("model.language_model.layers.{i}");
        let kind = base.layer_kind(i);
        let full = kind == LayerType::FullAttention;
        let hd = base.head_dim_for(kind);
        let n_kv = base.num_kv_heads_for(kind);
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "post_feedforward_layernorm_1",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_2",
        ] {
            t.insert(format!("{p}.{norm}.weight"), norm_tensor(&mut rng, HIDDEN));
        }
        t.insert(
            format!("{p}.layer_scalar"),
            Tensor::from_vec(vec![0.9f32 + 0.1 * rng.next_f32()], 1, &Device::Cpu).unwrap(),
        );
        t.insert(
            format!("{p}.self_attn.q_proj.weight"),
            rand_tensor(&mut rng, &[N_Q * hd, HIDDEN], 0.3),
        );
        t.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rand_tensor(&mut rng, &[n_kv * hd, HIDDEN], 0.3),
        );
        if !(full && base.attention_k_eq_v) {
            t.insert(
                format!("{p}.self_attn.v_proj.weight"),
                rand_tensor(&mut rng, &[n_kv * hd, HIDDEN], 0.3),
            );
        }
        t.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rand_tensor(&mut rng, &[HIDDEN, N_Q * hd], 0.3),
        );
        t.insert(
            format!("{p}.self_attn.q_norm.weight"),
            norm_tensor(&mut rng, hd),
        );
        t.insert(
            format!("{p}.self_attn.k_norm.weight"),
            norm_tensor(&mut rng, hd),
        );
        t.insert(
            format!("{p}.mlp.gate_proj.weight"),
            rand_tensor(&mut rng, &[INTER, HIDDEN], 0.3),
        );
        t.insert(
            format!("{p}.mlp.up_proj.weight"),
            rand_tensor(&mut rng, &[INTER, HIDDEN], 0.3),
        );
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            rand_tensor(&mut rng, &[HIDDEN, INTER], 0.3),
        );
        t.insert(
            format!("{p}.router.proj.weight"),
            rand_tensor(&mut rng, &[N_EXPERTS, HIDDEN], 0.3),
        );
        t.insert(format!("{p}.router.scale"), norm_tensor(&mut rng, HIDDEN));
        t.insert(
            format!("{p}.router.per_expert_scale"),
            norm_tensor(&mut rng, N_EXPERTS),
        );
        t.insert(
            format!("{p}.experts.gate_up_proj"),
            rand_tensor(&mut rng, &[N_EXPERTS, 2 * MOE_INTER, HIDDEN], 0.3),
        );
        t.insert(
            format!("{p}.experts.down_proj"),
            rand_tensor(&mut rng, &[N_EXPERTS, HIDDEN, MOE_INTER], 0.3),
        );
    }
    t
}

fn temp_dir(tag: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("g4m_i8route_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

fn lmhead_entries(m: &Gemma4MoeWgpu) -> Vec<String> {
    m.pass_rows()
        .into_iter()
        .filter(|(label, _, _, _)| label == "g4m-lmhead")
        .map(|(_, entry, _, _)| entry)
        .collect()
}

#[test]
fn quantize_i8_host_is_bit_exact_on_the_q_over_128_grid() {
    assert!(
        lmhead_i8_entry_off_until_real_checkpoint_quality_gate().is_none(),
        "the int8 lm-head must be OFF by default: it is not bit-identical to the bf16 head \
         and the real checkpoint's quality gate has not run"
    );
    let mut rng = Lcg(0x1d_5eed);
    let w = q_over_128_grid_embed(&mut rng, VOCAB, HIDDEN);
    let bits: Vec<u16> = w.iter().map(|v| bf16_bits(*v)).collect();
    for (i, v) in w.iter().enumerate() {
        assert_eq!(
            bf16_val(bits[i]),
            *v,
            "q/128 grid value {v} at {i} is not bf16-exact; the lossless premise is broken \
             before quantization even runs"
        );
    }
    let q = quantize_i8_host(&bits, VOCAB, HIDDEN);
    let mut nonconst_rows = 0usize;
    for r in 0..VOCAB {
        let deq = dequantize_i8_row(&q, r);
        for j in 0..HIDDEN {
            assert_eq!(
                deq[j],
                bf16_val(bits[r * HIDDEN + j]),
                "row {r} elem {j}: dequantized int8 differs from the bf16 source on the \
                 q/128 grid, so the routing golden test upstream of this cannot claim its \
                 two arms hold the same weights"
            );
        }
        if deq.iter().any(|v| *v != deq[0]) {
            nonconst_rows += 1;
        }
    }
    assert_eq!(
        nonconst_rows, VOCAB,
        "the engineered embed rows are degenerate; a constant matrix proves nothing"
    );
}

#[test]
fn gated_i8_lmhead_matches_the_bf16_head_on_lossless_weights() {
    if !have_gpu() {
        return;
    }
    assert!(
        std::env::var("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED").is_err(),
        "NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED is pre-set on the runner; this test owns that knob"
    );
    assert!(
        std::env::var("NV_G4MOE_ROW_CHUNK").is_err(),
        "NV_G4MOE_ROW_CHUNK is pre-set on the runner; this test owns that knob"
    );

    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let tensors = tiny_tensors_with_lossless_head(0x1d_5eed);
    let dir = temp_dir("st");
    let st = dir.0.join("model.safetensors");
    candle_core::safetensors::save(&tensors, &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();

    unsafe { std::env::set_var("NV_G4MOE_ROW_CHUNK", ROW_CHUNK.to_string()) };
    let build = |gate: Option<&str>| -> Gemma4MoeWgpu {
        match gate {
            Some(v) => unsafe { std::env::set_var("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED", v) },
            None => unsafe { std::env::remove_var("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED") },
        }
        let m = Gemma4MoeWgpu::from_loader(cfg.clone(), &loader, 16).expect("build");
        unsafe { std::env::remove_var("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED") };
        m
    };
    let mut bf16 = build(None);
    let mut i8s = build(Some("1"));
    let mut i8v4 = build(Some("v4"));
    unsafe { std::env::remove_var("NV_G4MOE_ROW_CHUNK") };

    let chunks = VOCAB.div_ceil(ROW_CHUNK);
    assert_eq!(
        lmhead_entries(&bf16),
        vec!["g4m_gemv_bf16"; chunks],
        "the default graph no longer dispatches the bf16 head; a default changed"
    );
    assert!(
        !bf16
            .pass_rows()
            .iter()
            .any(|(_, entry, _, _)| entry.contains("i8")),
        "an int8 entry is in the DEFAULT decode graph; the gate leaked"
    );
    assert_eq!(
        lmhead_entries(&i8s),
        vec!["g4m_gemv_i8"; chunks],
        "NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED=1 did not route g4m_gemv_i8 on every head chunk"
    );
    assert_eq!(
        lmhead_entries(&i8v4),
        vec!["g4m_gemv_i8_v4"; chunks],
        "NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED=v4 did not route g4m_gemv_i8_v4 on every head chunk"
    );
    for arm in [&i8s, &i8v4] {
        assert_eq!(
            arm.pass_count(),
            bf16.pass_count(),
            "the arms must differ only in which head entry runs, not in the dispatch list"
        );
    }
    let non_head = |m: &Gemma4MoeWgpu| -> Vec<(String, String)> {
        m.pass_rows()
            .into_iter()
            .filter(|(label, _, _, _)| label != "g4m-lmhead")
            .map(|(label, entry, _, _)| (label, entry))
            .collect()
    };
    assert_eq!(
        non_head(&bf16),
        non_head(&i8s),
        "a non-head pass differs between the arms; the gate touched more than the head"
    );

    let head_bf16_bytes = (VOCAB * HIDDEN * 2) as i64;
    let head_i8_bytes = (VOCAB * HIDDEN + VOCAB * (HIDDEN / I8_GS) * 2) as i64;
    let got = bf16.weight_bytes_per_token() as i64 - i8s.weight_bytes_per_token() as i64;
    assert_eq!(
        got,
        head_bf16_bytes - head_i8_bytes,
        "the per-token weight accounting does not show the head's bf16->int8 byte saving; \
         either the int8 buffers are uncounted or the gather-only embed is still charged \
         as head traffic"
    );

    let mut prev: Option<Vec<f32>> = None;
    let mut sensitive = false;
    for step in 0..STEPS {
        let fed = ((3 + 41 * step) % VOCAB) as u32;
        let (ta, la) = bf16.decode_step_logits(fed).expect("bf16 step");
        let (tb, lb) = i8s.decode_step_logits(fed).expect("i8 step");
        let (tc, lc) = i8v4.decode_step_logits(fed).expect("i8v4 step");
        let mag = la.iter().fold(0f32, |a, b| a.max(b.abs()));
        assert!(
            mag > 1e-3,
            "step {step}: the logit vector is degenerate (max |logit| {mag:.3e}); this \
             comparison would pass on zeros"
        );
        for (arm, l) in [("i8", &lb), ("i8v4", &lc)] {
            for (i, (a, b)) in la.iter().zip(l.iter()).enumerate() {
                let bound = 2.0 * BF16_ULP_REL * a.abs().max(b.abs()) + 1e-6;
                assert!(
                    (a - b).abs() <= bound,
                    "step {step} word {i}: {arm} head reads {b} where the bf16 head reads \
                     {a}, {:.3e} apart against a one-bf16-ulp bound of {bound:.3e}. On \
                     weights that quantize losslessly the ONLY admissible difference is \
                     the int8 kernel's per-word reassociation, so this arm is not \
                     computing the same head",
                    (a - b).abs()
                );
            }
        }
        assert_eq!(
            (ta, ta),
            (tb, tc),
            "step {step}: greedy tokens diverged (bf16 {ta}, i8 {tb}, i8v4 {tc}) while every \
             logit sat within one bf16 ulp"
        );
        if let Some(p) = &prev {
            if p.iter().zip(la.iter()).any(|(a, b)| a != b) {
                sensitive = true;
            }
        }
        prev = Some(la);
    }
    assert!(
        sensitive,
        "positive control failed: consecutive decode steps produced identical logits, so \
         this comparison cannot detect a head that computes something else"
    );
}
