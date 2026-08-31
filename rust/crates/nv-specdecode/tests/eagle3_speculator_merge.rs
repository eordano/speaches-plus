use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_models::eagle3_train::EAGLE3_HEAD_LORA_EXCLUDES_EMBED_AND_LM_HEAD_BECAUSE_THE_UPSTREAM_DRAFTER_RECIPE_FREEZES_BOTH as TRAINER_TARGETS;
use nv_specdecode::eagle3_loader::{
    cotrained_eagle3_speculator_dir, Eagle3SpeculatorConfig, LoadedEagle3Scorer,
    EAGLE3_SPECULATOR_DELTA_TARGETS_EXCLUDE_EMBED_AND_LM_HEAD_BECAUSE_THE_TRAINER_FREEZES_BOTH_AND_EMBED_MAY_BE_TENSOR_SHARED_WITH_THE_TARGET_TRUNK as DELTA_TARGETS,
    EAGLE3_SPECULATOR_KIND, SPECULATOR_SUBDIR,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const HIDDEN: usize = 32;
const INTER: usize = 64;
const N_Q: usize = 2;
const N_KV: usize = 1;
const HEAD_DIM: usize = 16;
const VOCAB: usize = 96;
const DRAFT_VOCAB: usize = 48;
const AUX_TAPS: usize = 3;

fn cfg() -> Eagle3SpeculatorConfig {
    Eagle3SpeculatorConfig {
        hidden_size: HIDDEN,
        draft_vocab_size: DRAFT_VOCAB,
        target_vocab_size: VOCAB,
        num_attention_heads: N_Q,
        num_key_value_heads: N_KV,
        head_dim: HEAD_DIM,
        intermediate_size: INTER,
        max_position_embeddings: 256,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        norm_before_residual: true,
        norm_before_fc: false,
        eagle_aux_hidden_state_layer_ids: vec![0, 1, 2],
    }
}

fn drafter_config_json() -> String {
    format!(
        r#"{{
  "speculators_model_type": "eagle3",
  "draft_vocab_size": {DRAFT_VOCAB},
  "norm_before_residual": true,
  "norm_before_fc": false,
  "eagle_aux_hidden_state_layer_ids": [0, 1, 2],
  "transformer_layer_config": {{
    "hidden_size": {HIDDEN},
    "intermediate_size": {INTER},
    "num_attention_heads": {N_Q},
    "num_key_value_heads": {N_KV},
    "head_dim": {HEAD_DIM},
    "max_position_embeddings": 256,
    "vocab_size": {VOCAB},
    "rms_norm_eps": 1e-6,
    "rope_parameters": {{"rope_theta": 10000.0}}
  }}
}}"#
    )
}

fn det_bf16(seed: usize, n: usize, scale: f32) -> Vec<bf16> {
    (0..n)
        .map(|i| bf16::from_f32(((seed * 31 + i) as f32 * 0.017).sin() * scale))
        .collect()
}

fn bf16_tensor(vals: &[bf16], shape: (usize, usize)) -> Tensor {
    Tensor::from_vec(vals.to_vec(), shape, &Device::Cpu).unwrap()
}

fn module_shapes() -> Vec<(&'static str, usize, usize)> {
    vec![
        ("fc", HIDDEN, AUX_TAPS * HIDDEN),
        ("layers.0.self_attn.q_proj", N_Q * HEAD_DIM, 2 * HIDDEN),
        ("layers.0.self_attn.k_proj", N_KV * HEAD_DIM, 2 * HIDDEN),
        ("layers.0.self_attn.v_proj", N_KV * HEAD_DIM, 2 * HIDDEN),
        ("layers.0.self_attn.o_proj", HIDDEN, N_Q * HEAD_DIM),
        ("layers.0.mlp.gate_proj", INTER, HIDDEN),
        ("layers.0.mlp.up_proj", INTER, HIDDEN),
        ("layers.0.mlp.down_proj", HIDDEN, INTER),
    ]
}

struct TinyDrafter {
    dir: PathBuf,
    host: HashMap<String, (Vec<bf16>, (usize, usize))>,
}

fn write_tiny_drafter(dir: &Path) -> TinyDrafter {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), drafter_config_json()).unwrap();
    let mut host: HashMap<String, (Vec<bf16>, (usize, usize))> = HashMap::new();
    host.insert(
        "embed_tokens".into(),
        (det_bf16(1, VOCAB * HIDDEN, 0.4), (VOCAB, HIDDEN)),
    );
    host.insert(
        "lm_head".into(),
        (det_bf16(2, DRAFT_VOCAB * HIDDEN, 0.3), (DRAFT_VOCAB, HIDDEN)),
    );
    for (i, (m, out_f, in_f)) in module_shapes().into_iter().enumerate() {
        host.insert(m.to_string(), (det_bf16(10 + i, out_f * in_f, 0.2), (out_f, in_f)));
    }
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "embed_tokens.weight".into(),
        bf16_tensor(&host["embed_tokens"].0, host["embed_tokens"].1),
    );
    t.insert(
        "lm_head.weight".into(),
        bf16_tensor(&host["lm_head"].0, host["lm_head"].1),
    );
    for (m, _, _) in module_shapes() {
        t.insert(format!("{m}.weight"), bf16_tensor(&host[m].0, host[m].1));
    }
    for norm in [
        "layers.0.input_layernorm",
        "layers.0.hidden_norm",
        "layers.0.post_attention_layernorm",
        "norm",
    ] {
        t.insert(
            format!("{norm}.weight"),
            Tensor::ones(HIDDEN, DType::BF16, &Device::Cpu).unwrap(),
        );
    }
    let d2t: Vec<i64> = (0..DRAFT_VOCAB as i64).collect();
    t.insert(
        "d2t".into(),
        Tensor::from_vec(d2t, DRAFT_VOCAB, &Device::Cpu).unwrap(),
    );
    let t2d: Vec<u8> = (0..VOCAB).map(|i| u8::from(i < DRAFT_VOCAB)).collect();
    t.insert(
        "t2d".into(),
        Tensor::from_vec(t2d, VOCAB, &Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&t, dir.join("model.safetensors")).unwrap();
    TinyDrafter {
        dir: dir.to_path_buf(),
        host,
    }
}

struct SpecEntry {
    module: String,
    a: Vec<bf16>,
    b: Vec<bf16>,
    a_shape: (usize, usize),
    b_shape: (usize, usize),
}

fn spec_entry(module: &str, r: usize, out_f: usize, in_f: usize, seed: usize, b_scale: f32) -> SpecEntry {
    SpecEntry {
        module: module.to_string(),
        a: det_bf16(100 + seed, r * in_f, 0.1),
        b: det_bf16(200 + seed, out_f * r, b_scale),
        a_shape: (r, in_f),
        b_shape: (out_f, r),
    }
}

fn speculator_config_json(
    entries: &[SpecEntry],
    r: usize,
    alpha: f64,
    kind: Option<&str>,
) -> String {
    let mut targets: Vec<String> = Vec::new();
    for e in entries {
        let leaf = e.module.rsplit('.').next().unwrap().to_string();
        if !targets.contains(&leaf) {
            targets.push(leaf);
        }
    }
    targets.sort();
    let mut v = serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "modules_to_save": serde_json::Value::Null,
        "fan_in_fan_out": false,
        "inference_mode": true,
        "target_modules": targets,
        "base_model_name_or_path": "drafter",
    });
    if let Some(k) = kind {
        v.as_object_mut()
            .unwrap()
            .insert("nv_speculator_kind".into(), serde_json::json!(k));
    }
    serde_json::to_string_pretty(&v).unwrap()
}

fn write_speculator(dir: &Path, entries: &[SpecEntry], r: usize, alpha: f64, kind: Option<&str>) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("adapter_config.json"),
        speculator_config_json(entries, r, alpha, kind),
    )
    .unwrap();
    let mut t: HashMap<String, Tensor> = HashMap::new();
    for e in entries {
        t.insert(
            format!("base_model.model.{}.lora_A.weight", e.module),
            bf16_tensor(&e.a, e.a_shape),
        );
        t.insert(
            format!("base_model.model.{}.lora_B.weight", e.module),
            bf16_tensor(&e.b, e.b_shape),
        );
    }
    candle_core::safetensors::save(&t, dir.join("adapter_model.safetensors")).unwrap();
}

fn fresh_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "nv-eagle3-spec-merge-{}-{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn forward_f32(scorer: &LoadedEagle3Scorer, seq: usize) -> Vec<f32> {
    let ctx: Vec<u32> = (0..seq as u32).map(|i| (i * 7 + 3) % VOCAB as u32).collect();
    let aux: Vec<f32> = (0..seq * AUX_TAPS * HIDDEN)
        .map(|i| ((i as f32) * 0.011).sin() * 0.3)
        .collect();
    let aux = Tensor::from_vec(aux, (seq, AUX_TAPS * HIDDEN), &Device::Cpu).unwrap();
    scorer
        .forward(&ctx, Some(&aux))
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn scorer_from_host(
    fixture: &TinyDrafter,
    merged_override: &HashMap<String, Vec<bf16>>,
) -> LoadedEagle3Scorer {
    let c = cfg();
    let eps = c.rms_norm_eps;
    let lin = |name: &str| -> Linear {
        let (base, shape) = &fixture.host[name];
        let vals = merged_override.get(name).unwrap_or(base);
        Linear::new(bf16_tensor(vals, *shape), None).unwrap()
    };
    let rms = || {
        RmsNorm::new(
            Tensor::ones(HIDDEN, DType::BF16, &Device::Cpu).unwrap(),
            eps,
        )
    };
    let d2t: Vec<u32> = (0..DRAFT_VOCAB as u32).collect();
    let t2d: Vec<bool> = (0..VOCAB).map(|i| i < DRAFT_VOCAB).collect();
    LoadedEagle3Scorer::from_parts(
        c,
        Device::Cpu,
        DType::BF16,
        bf16_tensor(&fixture.host["embed_tokens"].0, fixture.host["embed_tokens"].1),
        lin("fc"),
        rms(),
        rms(),
        rms(),
        lin("layers.0.self_attn.q_proj"),
        lin("layers.0.self_attn.k_proj"),
        lin("layers.0.self_attn.v_proj"),
        lin("layers.0.self_attn.o_proj"),
        lin("layers.0.mlp.gate_proj"),
        lin("layers.0.mlp.up_proj"),
        lin("layers.0.mlp.down_proj"),
        rms(),
        lin("lm_head"),
        d2t,
        t2d,
    )
    .unwrap()
}

fn full_rank1_entries(b_scale: f32) -> Vec<SpecEntry> {
    module_shapes()
        .into_iter()
        .enumerate()
        .map(|(i, (m, out_f, in_f))| spec_entry(m, 1, out_f, in_f, i, b_scale))
        .collect()
}

fn rank1_merged_host(
    fixture: &TinyDrafter,
    entries: &[SpecEntry],
    scaling: f32,
) -> HashMap<String, Vec<bf16>> {
    let mut out = HashMap::new();
    for e in entries {
        let (w, (out_f, in_f)) = &fixture.host[&e.module];
        let mut merged = Vec::with_capacity(w.len());
        for i in 0..*out_f {
            for j in 0..*in_f {
                let delta = e.b[i].to_f32() * e.a[j].to_f32() * scaling;
                merged.push(bf16::from_f32(w[i * in_f + j].to_f32() + delta));
            }
        }
        out.insert(e.module.clone(), merged);
    }
    out
}

#[test]
fn without_a_speculator_dir_the_loaded_drafter_is_the_plain_try_load_drafter_bit_for_bit() {
    let root = fresh_dir("plain");
    let fixture = write_tiny_drafter(&root.join("drafter"));
    let plain = LoadedEagle3Scorer::try_load(&fixture.dir, &Device::Cpu).unwrap();
    let with_none =
        LoadedEagle3Scorer::try_load_with_speculator_deltas(&fixture.dir, None, &Device::Cpu)
            .unwrap();
    assert_eq!(
        forward_f32(&plain, 6),
        forward_f32(&with_none, 6),
        "a None speculator dir must leave the load path byte-identical to try_load"
    );
}

#[test]
fn zero_b_deltas_merge_to_a_value_identical_drafter() {
    let root = fresh_dir("zerob");
    let fixture = write_tiny_drafter(&root.join("drafter"));
    let entries: Vec<SpecEntry> = full_rank1_entries(0.0)
        .into_iter()
        .map(|mut e| {
            e.b = vec![bf16::from_f32(0.0); e.b.len()];
            e
        })
        .collect();
    let sdir = root.join(SPECULATOR_SUBDIR);
    write_speculator(&sdir, &entries, 1, 1.0, None);
    let plain = LoadedEagle3Scorer::try_load(&fixture.dir, &Device::Cpu).unwrap();
    let merged =
        LoadedEagle3Scorer::try_load_with_speculator_deltas(&fixture.dir, Some(&sdir), &Device::Cpu)
            .unwrap();
    assert_eq!(
        forward_f32(&plain, 6),
        forward_f32(&merged, 6),
        "an untrained (all-zero B) speculator must merge to the identical drafter function"
    );
}

#[test]
fn rank1_deltas_merge_per_the_f32_recipe_with_one_bf16_rounding_at_the_end() {
    let root = fresh_dir("rank1");
    let fixture = write_tiny_drafter(&root.join("drafter"));
    let entries = full_rank1_entries(0.15);
    let sdir = root.join(SPECULATOR_SUBDIR);
    write_speculator(&sdir, &entries, 1, 1.0, Some(EAGLE3_SPECULATOR_KIND));
    let merged =
        LoadedEagle3Scorer::try_load_with_speculator_deltas(&fixture.dir, Some(&sdir), &Device::Cpu)
            .unwrap();
    let expected = scorer_from_host(&fixture, &rank1_merged_host(&fixture, &entries, 1.0));
    assert_eq!(
        forward_f32(&expected, 6),
        forward_f32(&merged, 6),
        "merged forward must match the scalar oracle w' = bf16(f32(w) + f32(B)*f32(A))"
    );
    let plain = LoadedEagle3Scorer::try_load(&fixture.dir, &Device::Cpu).unwrap();
    assert_ne!(
        forward_f32(&plain, 6),
        forward_f32(&merged, 6),
        "nonzero deltas must change the drafter, otherwise this test verifies nothing"
    );
}

#[test]
fn alpha_unequal_r_scales_the_delta_in_f32_never_by_folding_into_bf16_b() {
    let root = fresh_dir("alpha");
    let fixture = write_tiny_drafter(&root.join("drafter"));
    let entries = full_rank1_entries(0.15);
    let sdir = root.join(SPECULATOR_SUBDIR);
    write_speculator(&sdir, &entries, 1, 3.0, None);
    let merged =
        LoadedEagle3Scorer::try_load_with_speculator_deltas(&fixture.dir, Some(&sdir), &Device::Cpu)
            .unwrap();
    let expected = scorer_from_host(&fixture, &rank1_merged_host(&fixture, &entries, 3.0));
    assert_eq!(
        forward_f32(&expected, 6),
        forward_f32(&merged, 6),
        "lora_alpha != r must scale the f32 delta (alpha/r) before the single bf16 rounding; \
         a LoraAdapter::optimize-style bf16 fold of scaling into B would diverge here"
    );
}

#[test]
fn a_delta_naming_embed_tokens_lm_head_a_norm_or_d2t_refuses_the_whole_merge() {
    let root = fresh_dir("refuse");
    let fixture = write_tiny_drafter(&root.join("drafter"));
    for (tag, bad_module, out_f, in_f) in [
        ("embed", "embed_tokens", VOCAB, HIDDEN),
        ("lmhead", "lm_head", DRAFT_VOCAB, HIDDEN),
        ("norm", "layers.0.input_layernorm", HIDDEN, HIDDEN),
    ] {
        let mut entries = vec![spec_entry("fc", 1, HIDDEN, AUX_TAPS * HIDDEN, 0, 0.15)];
        entries.push(spec_entry(bad_module, 1, out_f, in_f, 9, 0.15));
        let sdir = root.join(format!("spec-{tag}"));
        write_speculator(&sdir, &entries, 1, 1.0, None);
        let mut scorer = LoadedEagle3Scorer::try_load(&fixture.dir, &Device::Cpu).unwrap();
        let before = forward_f32(&scorer, 6);
        let err = scorer
            .merge_speculator_deltas(&sdir)
            .expect_err("a non-target delta module must refuse the merge");
        assert!(
            format!("{err:#}").contains(bad_module.split('.').next_back().unwrap())
                || format!("{err:#}").contains(bad_module),
            "refusal must name the offending module, got: {err:#}"
        );
        assert_eq!(
            before,
            forward_f32(&scorer, 6),
            "a refused merge must not partially apply the valid fc delta ({tag})"
        );
    }
    let sdir = root.join("spec-d2t");
    write_speculator(&sdir, &[spec_entry("fc", 1, HIDDEN, AUX_TAPS * HIDDEN, 0, 0.15)], 1, 1.0, None);
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "base_model.model.fc.lora_A.weight".into(),
        bf16_tensor(&det_bf16(0, AUX_TAPS * HIDDEN, 0.1), (1, AUX_TAPS * HIDDEN)),
    );
    t.insert(
        "base_model.model.fc.lora_B.weight".into(),
        bf16_tensor(&det_bf16(1, HIDDEN, 0.1), (HIDDEN, 1)),
    );
    t.insert(
        "d2t".into(),
        Tensor::from_vec((0..DRAFT_VOCAB as i64).collect::<Vec<i64>>(), DRAFT_VOCAB, &Device::Cpu)
            .unwrap(),
    );
    candle_core::safetensors::save(&t, sdir.join("adapter_model.safetensors")).unwrap();
    let mut scorer = LoadedEagle3Scorer::try_load(&fixture.dir, &Device::Cpu).unwrap();
    let before = forward_f32(&scorer, 6);
    let err = scorer
        .merge_speculator_deltas(&sdir)
        .expect_err("a stray d2t tensor in the speculator pair must refuse the merge");
    assert!(
        format!("{err:#}").contains("d2t"),
        "refusal must name the stray tensor, got: {err:#}"
    );
    assert_eq!(before, forward_f32(&scorer, 6));
}

#[test]
fn mismatched_in_features_mean_a_different_drafter_and_refuse_without_partial_merge() {
    let root = fresh_dir("shape");
    let fixture = write_tiny_drafter(&root.join("drafter"));
    let entries = vec![
        spec_entry("layers.0.self_attn.q_proj", 1, N_Q * HEAD_DIM, 2 * HIDDEN, 1, 0.15),
        spec_entry("fc", 1, HIDDEN, 2 * HIDDEN, 0, 0.15),
    ];
    let sdir = root.join(SPECULATOR_SUBDIR);
    write_speculator(&sdir, &entries, 1, 1.0, None);
    let mut scorer = LoadedEagle3Scorer::try_load(&fixture.dir, &Device::Cpu).unwrap();
    let before = forward_f32(&scorer, 6);
    let err = scorer
        .merge_speculator_deltas(&sdir)
        .expect_err("an fc delta trained for a two-tap drafter must refuse against three taps");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("fc") && msg.contains("different drafter"),
        "refusal must name the module and the drafter mismatch, got: {msg}"
    );
    assert_eq!(
        before,
        forward_f32(&scorer, 6),
        "the valid q_proj delta must not be applied when fc refuses"
    );
}

#[test]
fn an_incomplete_lora_pair_is_refused() {
    let root = fresh_dir("halfpair");
    let fixture = write_tiny_drafter(&root.join("drafter"));
    let sdir = root.join(SPECULATOR_SUBDIR);
    write_speculator(&sdir, &[spec_entry("fc", 1, HIDDEN, AUX_TAPS * HIDDEN, 0, 0.15)], 1, 1.0, None);
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "base_model.model.layers.0.mlp.gate_proj.lora_A.weight".into(),
        bf16_tensor(&det_bf16(3, HIDDEN, 0.1), (1, HIDDEN)),
    );
    candle_core::safetensors::save(&t, sdir.join("adapter_model.safetensors")).unwrap();
    let mut scorer = LoadedEagle3Scorer::try_load(&fixture.dir, &Device::Cpu).unwrap();
    let err = scorer
        .merge_speculator_deltas(&sdir)
        .expect_err("lora_A without lora_B must refuse");
    assert!(
        format!("{err:#}").contains("incomplete"),
        "got: {err:#}"
    );
}

#[test]
fn a_foreign_kind_in_the_subdir_config_is_refused_naming_both_kinds() {
    let root = fresh_dir("kind");
    let fixture = write_tiny_drafter(&root.join("drafter"));
    let entries = full_rank1_entries(0.15);
    let sdir = root.join(SPECULATOR_SUBDIR);
    write_speculator(&sdir, &entries, 1, 1.0, Some("qwen38_mtp_head"));
    let mut scorer = LoadedEagle3Scorer::try_load(&fixture.dir, &Device::Cpu).unwrap();
    let err = scorer
        .merge_speculator_deltas(&sdir)
        .expect_err("an MTP-head speculator pair must not merge into the eagle3 drafter");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("qwen38_mtp_head") && msg.contains(EAGLE3_SPECULATOR_KIND),
        "refusal must name both kinds, got: {msg}"
    );
}

#[test]
fn share_embed_tokens_with_target_still_holds_after_a_delta_merge() {
    let root = fresh_dir("shareembed");
    let fixture = write_tiny_drafter(&root.join("drafter"));
    let entries = full_rank1_entries(0.15);
    let sdir = root.join(SPECULATOR_SUBDIR);
    write_speculator(&sdir, &entries, 1, 1.0, None);
    let mut scorer = LoadedEagle3Scorer::try_load(&fixture.dir, &Device::Cpu).unwrap();
    let target_embed = bf16_tensor(&fixture.host["embed_tokens"].0, fixture.host["embed_tokens"].1);
    assert!(
        scorer.share_embed_tokens_with_target(&target_embed).unwrap(),
        "fixture embed must dedup against an equal target embed"
    );
    let before = forward_f32(&scorer, 6);
    scorer.merge_speculator_deltas(&sdir).unwrap();
    assert!(
        scorer.share_embed_tokens_with_target(&target_embed).unwrap(),
        "the merge must never write embed_tokens: a delta there would mutate the tensor the \
         target trunk shares"
    );
    assert_ne!(
        before,
        forward_f32(&scorer, 6),
        "the merge must have applied the projection deltas while leaving embed shared"
    );
}

#[test]
fn discovery_reads_the_main_config_kind_and_requires_the_advertised_pair() {
    let root = fresh_dir("discovery");
    let adapter = root.join("adapter");
    std::fs::create_dir_all(&adapter).unwrap();
    let write_main = |kind: Option<&str>| {
        let mut v = serde_json::json!({"r": 4, "lora_alpha": 4.0, "target_modules": ["q_proj"]});
        if let Some(k) = kind {
            v.as_object_mut()
                .unwrap()
                .insert("nv_speculator_kind".into(), serde_json::json!(k));
        }
        std::fs::write(
            adapter.join("adapter_config.json"),
            serde_json::to_string_pretty(&v).unwrap(),
        )
        .unwrap();
    };
    write_main(None);
    assert_eq!(
        cotrained_eagle3_speculator_dir(&adapter).unwrap(),
        None,
        "no kind key means no co-trained speculator"
    );
    write_main(Some("qwen38_mtp_head"));
    assert_eq!(
        cotrained_eagle3_speculator_dir(&adapter).unwrap(),
        None,
        "an MTP-head co-trained artifact carries no eagle3 deltas"
    );
    write_main(Some(EAGLE3_SPECULATOR_KIND));
    let err = cotrained_eagle3_speculator_dir(&adapter)
        .expect_err("an advertised but missing speculator/ pair must refuse");
    assert!(format!("{err:#}").contains("missing"), "got: {err:#}");
    let entries = full_rank1_entries(0.15);
    write_speculator(&adapter.join(SPECULATOR_SUBDIR), &entries, 1, 1.0, None);
    assert_eq!(
        cotrained_eagle3_speculator_dir(&adapter).unwrap(),
        Some(adapter.join(SPECULATOR_SUBDIR))
    );
}

#[test]
fn an_empty_speculator_pair_is_refused() {
    let root = fresh_dir("empty");
    let fixture = write_tiny_drafter(&root.join("drafter"));
    let sdir = root.join(SPECULATOR_SUBDIR);
    write_speculator(&sdir, &[], 1, 1.0, None);
    let mut scorer = LoadedEagle3Scorer::try_load(&fixture.dir, &Device::Cpu).unwrap();
    let err = scorer
        .merge_speculator_deltas(&sdir)
        .expect_err("zero lora pairs must refuse");
    assert!(
        format!("{err:#}").contains("no lora_A/lora_B pairs"),
        "got: {err:#}"
    );
}

#[test]
fn the_delta_target_list_matches_the_trainer_head_module_names() {
    let mut trainer: Vec<&str> = TRAINER_TARGETS.to_vec();
    trainer.sort_unstable();
    let mut loader: Vec<&str> = DELTA_TARGETS.to_vec();
    loader.sort_unstable();
    assert_eq!(
        loader, trainer,
        "the loader's delta target list must match the trainer's lora module list \
         module-for-module, or a trainer-side rename ships artifacts the loader refuses"
    );
    let shapes = module_shapes();
    assert_eq!(DELTA_TARGETS.len(), shapes.len());
    for (m, _, _) in shapes {
        assert!(
            DELTA_TARGETS.contains(&m),
            "fixture module {m} missing from the loader's delta target list"
        );
    }
    assert!(!DELTA_TARGETS.contains(&"embed_tokens"));
    assert!(!DELTA_TARGETS.contains(&"lm_head"));
}

#[test]
fn a_valid_delta_iterating_before_the_refusing_module_is_still_not_applied() {
    let root = fresh_dir("stagedorder");
    let fixture = write_tiny_drafter(&root.join("drafter"));
    let entries = vec![
        spec_entry("fc", 1, HIDDEN, AUX_TAPS * HIDDEN, 0, 0.15),
        spec_entry("layers.0.self_attn.o_proj", 1, HIDDEN, 2 * HIDDEN, 1, 0.15),
    ];
    let sdir = root.join(SPECULATOR_SUBDIR);
    write_speculator(&sdir, &entries, 1, 1.0, None);
    let mut scorer = LoadedEagle3Scorer::try_load(&fixture.dir, &Device::Cpu).unwrap();
    let before = forward_f32(&scorer, 6);
    let err = scorer
        .merge_speculator_deltas(&sdir)
        .expect_err("a shape-mismatched o_proj must refuse the merge");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("o_proj") && msg.contains("different drafter"),
        "refusal must name the module and the drafter mismatch, got: {msg}"
    );
    assert_eq!(
        before,
        forward_f32(&scorer, 6),
        "fc sorts before the refusing o_proj, so an eager per-module apply would have \
         mutated it; staged validation must leave the drafter untouched"
    );
}

#[test]
fn a_prefixed_and_unprefixed_spelling_of_one_lora_slot_refuses_instead_of_silently_picking() {
    let root = fresh_dir("aliasspelling");
    let fixture = write_tiny_drafter(&root.join("drafter"));
    let sdir = root.join(SPECULATOR_SUBDIR);
    write_speculator(
        &sdir,
        &[spec_entry("fc", 1, HIDDEN, AUX_TAPS * HIDDEN, 0, 0.15)],
        1,
        1.0,
        None,
    );
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "base_model.model.fc.lora_A.weight".into(),
        bf16_tensor(&det_bf16(0, AUX_TAPS * HIDDEN, 0.1), (1, AUX_TAPS * HIDDEN)),
    );
    t.insert(
        "fc.lora_A.weight".into(),
        bf16_tensor(&det_bf16(5, AUX_TAPS * HIDDEN, 0.1), (1, AUX_TAPS * HIDDEN)),
    );
    t.insert(
        "base_model.model.fc.lora_B.weight".into(),
        bf16_tensor(&det_bf16(1, HIDDEN, 0.1), (HIDDEN, 1)),
    );
    candle_core::safetensors::save(&t, sdir.join("adapter_model.safetensors")).unwrap();
    let mut scorer = LoadedEagle3Scorer::try_load(&fixture.dir, &Device::Cpu).unwrap();
    let before = forward_f32(&scorer, 6);
    let err = scorer
        .merge_speculator_deltas(&sdir)
        .expect_err("two spellings collapsing onto one lora_A slot must refuse");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("fc") && msg.contains("spelling"),
        "refusal must name the colliding module, got: {msg}"
    );
    assert_eq!(
        before,
        forward_f32(&scorer, 6),
        "a spelling-collision refusal must not partially apply anything"
    );
}

#[test]
fn a_non_string_speculator_kind_is_refused_not_read_as_absent() {
    let root = fresh_dir("nonstringkind");
    let adapter = root.join("adapter");
    std::fs::create_dir_all(&adapter).unwrap();
    std::fs::write(
        adapter.join("adapter_config.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "r": 4, "lora_alpha": 4.0, "target_modules": ["q_proj"], "nv_speculator_kind": 7
        }))
        .unwrap(),
    )
    .unwrap();
    let err = cotrained_eagle3_speculator_dir(&adapter)
        .expect_err("a numeric nv_speculator_kind in the main config must refuse discovery");
    assert!(
        format!("{err:#}").contains("non-string"),
        "got: {err:#}"
    );

    let fixture = write_tiny_drafter(&root.join("drafter"));
    let sdir = root.join(SPECULATOR_SUBDIR);
    let entries = full_rank1_entries(0.15);
    write_speculator(&sdir, &entries, 1, 1.0, None);
    let raw = std::fs::read_to_string(sdir.join("adapter_config.json")).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    v.as_object_mut()
        .unwrap()
        .insert("nv_speculator_kind".into(), serde_json::json!(7));
    std::fs::write(
        sdir.join("adapter_config.json"),
        serde_json::to_string_pretty(&v).unwrap(),
    )
    .unwrap();
    let mut scorer = LoadedEagle3Scorer::try_load(&fixture.dir, &Device::Cpu).unwrap();
    let before = forward_f32(&scorer, 6);
    let err = scorer
        .merge_speculator_deltas(&sdir)
        .expect_err("a numeric nv_speculator_kind in the subdir config must refuse the merge");
    assert!(
        format!("{err:#}").contains("non-string"),
        "got: {err:#}"
    );
    assert_eq!(before, forward_f32(&scorer, 6));
}
