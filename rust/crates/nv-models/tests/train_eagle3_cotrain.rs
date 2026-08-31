use candle_core::{DType, Device, Tensor};
use nv_models::eagle3_train::{Eagle3TrainConfig, Eagle3TrainHead};
use nv_models::train_runner::{
    load_base, run, run_with_speculator, BaseModel, DenseAuxTapForward, SpeculatorSpec, TrainArgs,
};
use nv_train::max_abs_diff;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const HIDDEN: usize = 32;
const INTER: usize = 64;
const N_LAYERS: usize = 3;
const N_Q: usize = 2;
const N_KV: usize = 1;
const HEAD_DIM: usize = 16;
const VOCAB: usize = 96;
const DRAFT_VOCAB: usize = 48;
const AUX_IDS: [usize; 3] = [0, 1, 2];

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 40) as u32;
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    fn token(&mut self) -> u32 {
        ((self.next_f32().abs() * VOCAB as f32) as u32) % VOCAB as u32
    }
}

fn rand_tensor(rng: &mut Lcg, shape: (usize, usize), scale: f32) -> Tensor {
    let n = shape.0 * shape.1;
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32() * scale).collect();
    Tensor::from_vec(data, shape, &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn ones(dim: usize) -> Tensor {
    Tensor::ones(dim, DType::BF16, &Device::Cpu).unwrap()
}

fn base_config_json() -> String {
    format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "hidden_size": {HIDDEN},
  "intermediate_size": {INTER},
  "num_hidden_layers": {N_LAYERS},
  "num_attention_heads": {N_Q},
  "num_key_value_heads": {N_KV},
  "num_global_key_value_heads": {N_KV},
  "head_dim": {HEAD_DIM},
  "global_head_dim": {HEAD_DIM},
  "vocab_size": {VOCAB},
  "max_position_embeddings": 256,
  "rms_norm_eps": 1e-6,
  "sliding_window": 8,
  "layer_types": ["full_attention", "sliding_attention", "full_attention"],
  "attention_k_eq_v": false,
  "tie_word_embeddings": false,
  "hidden_activation": "gelu_pytorch_tanh",
  "rope_parameters": {{
    "full_attention": {{"rope_theta": 10000.0, "partial_rotary_factor": 1.0}},
    "sliding_attention": {{"rope_theta": 10000.0}}
  }}
}}"#
    )
}

fn write_tiny_base(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), base_config_json()).unwrap();
    let mut rng = Lcg(0x5eed_cafe_f00d_0088);
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "model.language_model.embed_tokens.weight".into(),
        rand_tensor(&mut rng, (VOCAB, HIDDEN), 1.0),
    );
    t.insert("model.language_model.norm.weight".into(), ones(HIDDEN));
    t.insert(
        "lm_head.weight".into(),
        rand_tensor(&mut rng, (VOCAB, HIDDEN), 1.0),
    );
    for i in 0..N_LAYERS {
        let p = format!("model.language_model.layers.{i}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            t.insert(format!("{p}.{norm}.weight"), ones(HIDDEN));
        }
        t.insert(format!("{p}.layer_scalar"), ones(1));
        t.insert(
            format!("{p}.self_attn.q_proj.weight"),
            rand_tensor(&mut rng, (N_Q * HEAD_DIM, HIDDEN), 0.3),
        );
        t.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rand_tensor(&mut rng, (N_KV * HEAD_DIM, HIDDEN), 0.3),
        );
        t.insert(
            format!("{p}.self_attn.v_proj.weight"),
            rand_tensor(&mut rng, (N_KV * HEAD_DIM, HIDDEN), 0.3),
        );
        t.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rand_tensor(&mut rng, (HIDDEN, N_Q * HEAD_DIM), 0.3),
        );
        t.insert(format!("{p}.self_attn.q_norm.weight"), ones(HEAD_DIM));
        t.insert(format!("{p}.self_attn.k_norm.weight"), ones(HEAD_DIM));
        t.insert(
            format!("{p}.mlp.gate_proj.weight"),
            rand_tensor(&mut rng, (INTER, HIDDEN), 0.3),
        );
        t.insert(
            format!("{p}.mlp.up_proj.weight"),
            rand_tensor(&mut rng, (INTER, HIDDEN), 0.3),
        );
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            rand_tensor(&mut rng, (HIDDEN, INTER), 0.3),
        );
    }
    candle_core::safetensors::save(&t, dir.join("model.safetensors")).unwrap();
}

fn drafter_config_json(hidden: usize, target_vocab: usize, aux_ids: &[usize], kind: &str) -> String {
    let aux: Vec<String> = aux_ids.iter().map(|i| i.to_string()).collect();
    format!(
        r#"{{
  "speculators_model_type": "{kind}",
  "draft_vocab_size": {DRAFT_VOCAB},
  "norm_before_residual": true,
  "norm_before_fc": false,
  "eagle_aux_hidden_state_layer_ids": [{}],
  "transformer_layer_config": {{
    "hidden_size": {hidden},
    "intermediate_size": {INTER},
    "num_attention_heads": {N_Q},
    "num_key_value_heads": {N_KV},
    "head_dim": {HEAD_DIM},
    "max_position_embeddings": 256,
    "vocab_size": {target_vocab},
    "rms_norm_eps": 1e-6,
    "rope_parameters": {{"rope_theta": 10000.0}}
  }}
}}"#,
        aux.join(", ")
    )
}

fn write_tiny_drafter(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        drafter_config_json(HIDDEN, VOCAB, &AUX_IDS, "eagle3"),
    )
    .unwrap();
    let mut rng = Lcg(0xdead_beef_0042_0001);
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "embed_tokens.weight".into(),
        rand_tensor(&mut rng, (VOCAB, HIDDEN), 1.0),
    );
    t.insert(
        "fc.weight".into(),
        rand_tensor(&mut rng, (HIDDEN, AUX_IDS.len() * HIDDEN), 0.3),
    );
    for norm in ["input_layernorm", "hidden_norm", "post_attention_layernorm"] {
        t.insert(format!("layers.0.{norm}.weight"), ones(HIDDEN));
    }
    t.insert(
        "layers.0.self_attn.q_proj.weight".into(),
        rand_tensor(&mut rng, (N_Q * HEAD_DIM, 2 * HIDDEN), 0.3),
    );
    t.insert(
        "layers.0.self_attn.k_proj.weight".into(),
        rand_tensor(&mut rng, (N_KV * HEAD_DIM, 2 * HIDDEN), 0.3),
    );
    t.insert(
        "layers.0.self_attn.v_proj.weight".into(),
        rand_tensor(&mut rng, (N_KV * HEAD_DIM, 2 * HIDDEN), 0.3),
    );
    t.insert(
        "layers.0.self_attn.o_proj.weight".into(),
        rand_tensor(&mut rng, (HIDDEN, N_Q * HEAD_DIM), 0.3),
    );
    t.insert(
        "layers.0.mlp.gate_proj.weight".into(),
        rand_tensor(&mut rng, (INTER, HIDDEN), 0.3),
    );
    t.insert(
        "layers.0.mlp.up_proj.weight".into(),
        rand_tensor(&mut rng, (INTER, HIDDEN), 0.3),
    );
    t.insert(
        "layers.0.mlp.down_proj.weight".into(),
        rand_tensor(&mut rng, (HIDDEN, INTER), 0.3),
    );
    t.insert("norm.weight".into(), ones(HIDDEN));
    t.insert(
        "lm_head.weight".into(),
        rand_tensor(&mut rng, (DRAFT_VOCAB, HIDDEN), 0.3),
    );
    let d2t: Vec<i64> = (0..DRAFT_VOCAB as i64).collect();
    t.insert(
        "d2t".into(),
        Tensor::from_vec(d2t, DRAFT_VOCAB, &Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&t, dir.join("model.safetensors")).unwrap();
}

fn write_jsonl(path: &Path, rows: usize, seq: usize, seed: u64) {
    let mut rng = Lcg(seed);
    let mut out = String::new();
    for _ in 0..rows {
        let ids: Vec<String> = (0..seq).map(|_| rng.token().to_string()).collect();
        out.push_str(&format!("{{\"ids\": [{}]}}\n", ids.join(", ")));
    }
    std::fs::write(path, out).unwrap();
}

static FIXTURE: OnceLock<PathBuf> = OnceLock::new();

fn fixture_dir() -> &'static Path {
    FIXTURE.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("nv-eagle3-cotrain-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_tiny_base(&dir.join("base"));
        write_tiny_drafter(&dir.join("drafter"));
        write_jsonl(&dir.join("train.jsonl"), 2, 12, 0xfeed_0087);
        dir
    })
}

fn train_args(out: &Path) -> TrainArgs {
    let f = fixture_dir();
    TrainArgs {
        base: f.join("base"),
        data: f.join("train.jsonl"),
        eval_data: None,
        out: out.to_path_buf(),
        rank: 4,
        alpha: 8.0,
        targets: TrainArgs::default_targets(),
        steps: 2,
        lr: 1e-2,
        seed: 7,
    }
}

fn spec_with_rank_4() -> SpeculatorSpec {
    let mut s = SpeculatorSpec::eagle3_medusa2_defaults_rank_32_lr_ratio_4(
        fixture_dir().join("drafter"),
    );
    s.rank = 4;
    s
}

#[test]
fn tap_replica_logits_match_the_trainer_forward_bitwise() {
    let base = fixture_dir().join("base");
    let (model, base_dtype) = load_base(&base, &Device::Cpu).unwrap();
    let dense = match &model {
        BaseModel::Dense(d) => d,
        _ => panic!("tiny fixture must load as a dense base"),
    };
    let taps = DenseAuxTapForward::open(&base, dense, base_dtype).unwrap();
    let mut rng = Lcg(0xabcd_0087);
    let ids: Vec<u32> = (0..12).map(|_| rng.token()).collect();
    let (aux, teacher) = taps
        .taps_and_teacher_logits_detached(dense, &ids, &AUX_IDS)
        .unwrap();
    assert_eq!(
        aux.dims(),
        &[12, AUX_IDS.len() * HIDDEN],
        "aux rows must be the tap-order concat of the tapped layer outputs"
    );
    let tokens = Tensor::from_vec(ids.clone(), (1usize, 12usize), &Device::Cpu).unwrap();
    let positions = Tensor::from_vec((0..12u32).collect::<Vec<_>>(), 12, &Device::Cpu).unwrap();
    let oracle = model
        .forward_logits(&tokens, &positions)
        .unwrap()
        .squeeze(0)
        .unwrap();
    let diff = max_abs_diff(&teacher, &oracle).unwrap();
    assert_eq!(
        diff, 0.0,
        "the tap replica re-runs the exact op sequence of DenseTrainModel::forward_logits, so \
         any nonzero difference means the replica trains the drafter against a different \
         teacher than the target being trained (got max abs diff {diff:.3e})"
    );
}

#[test]
fn cotraining_leaves_the_target_adapter_bitwise_identical_and_ships_the_speculator_subdir() {
    let f = fixture_dir();
    let out_plain = f.join("out-plain");
    let out_cotrain = f.join("out-cotrain");

    let summary_plain = run(&train_args(&out_plain)).unwrap();
    let (summary_co, spec_summary) =
        run_with_speculator(&train_args(&out_cotrain), Some(&spec_with_rank_4())).unwrap();

    assert_eq!(
        summary_plain.losses, summary_co.losses,
        "target losses diverged, so the speculator branch leaked into target training"
    );
    let bytes_plain = std::fs::read(out_plain.join("adapter_model.safetensors")).unwrap();
    let bytes_co = std::fs::read(out_cotrain.join("adapter_model.safetensors")).unwrap();
    assert_eq!(
        bytes_plain, bytes_co,
        "the target adapter must be byte-identical with and without co-training: the taps \
         feeding the drafter are detached and the drafter has its own optimizer, so any byte \
         difference means speculator gradients or rng reached the target LoRAs"
    );

    let spec_summary = spec_summary.expect("co-training must return a speculator summary");
    assert_eq!(spec_summary.kind, "eagle3");
    assert_eq!(
        spec_summary.losses.len(),
        2,
        "one speculator distillation step must run per target step"
    );
    assert!(
        spec_summary.all_grads_finite,
        "speculator grads went non-finite on the tiny fixture"
    );
    assert!(
        spec_summary.losses.iter().all(|l| l.is_finite()),
        "speculator losses must be finite, got {:?}",
        spec_summary.losses
    );
    assert_eq!(spec_summary.trainable_vars, 16);

    let spec_dir = out_cotrain.join("speculator");
    let names: Vec<String> = std::fs::read_dir(&spec_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        vec!["adapter_config.json", "adapter_model.safetensors"],
        "speculator/ must hold exactly the peft pair so probe_adapter-style discovery works"
    );

    let main_cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out_cotrain.join("adapter_config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        main_cfg.get("nv_speculator_kind").and_then(|v| v.as_str()),
        Some("eagle3"),
        "the main adapter_config.json must name the co-trained speculator kind so the pair \
         ships and hot-swaps as one artifact"
    );
    let plain_cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(out_plain.join("adapter_config.json")).unwrap(),
    )
    .unwrap();
    assert!(
        plain_cfg.get("nv_speculator_kind").is_none(),
        "without co-training the config must be byte-identical to today, so no speculator field"
    );

    let spec_cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(spec_dir.join("adapter_config.json")).unwrap())
            .unwrap();
    let modules: Vec<String> = spec_cfg["target_modules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(
        !modules.iter().any(|m| m.contains("embed") || m.contains("lm_head")),
        "drafter LoRA must exclude embed_tokens and lm_head, got target_modules {modules:?}"
    );

    let tensors =
        candle_core::safetensors::load(spec_dir.join("adapter_model.safetensors"), &Device::Cpu)
            .unwrap();
    assert_eq!(spec_summary.modules.len(), 8);
    for m in &spec_summary.modules {
        let a = tensors
            .get(&format!("base_model.model.{m}.lora_A.weight"))
            .unwrap_or_else(|| panic!("missing lora_A for {m}"));
        let b = tensors
            .get(&format!("base_model.model.{m}.lora_B.weight"))
            .unwrap_or_else(|| panic!("missing lora_B for {m}"));
        assert_eq!(a.dims()[0], 4);
        let b_max: f32 = b
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            b_max > 0.0,
            "lora_B for {m} is still exactly zero after 2 distillation steps: no gradient \
             reached this drafter module"
        );
    }
}

fn fixture_head_f32() -> (Eagle3TrainConfig, Eagle3TrainHead) {
    let dir = fixture_dir().join("drafter");
    let cfg = Eagle3TrainConfig::from_drafter_dir(&dir).unwrap();
    let head = Eagle3TrainHead::load(&cfg, &dir, &Device::Cpu, DType::F32).unwrap();
    (cfg, head)
}

#[test]
fn shift_forward_never_reads_token_zero_and_row_i_reads_token_i_plus_1_and_aux_row_i() {
    let (cfg, head) = fixture_head_f32();
    let seq = 8usize;
    let fc_in = cfg.fc_in_dim();
    let mut rng = Lcg(0x51f7_0087);
    let ids: Vec<u32> = (0..seq).map(|_| rng.token()).collect();
    let aux_data: Vec<f32> = (0..seq * fc_in).map(|_| rng.next_f32()).collect();
    let aux_of = |data: Vec<f32>| Tensor::from_vec(data, (seq, fc_in), &Device::Cpu).unwrap();
    let fwd = |ids: &[u32], aux: &Tensor| {
        head.forward_train_shift_pairs_embed_of_token_i_plus_1_with_aux_row_i_like_serving(ids, aux)
            .unwrap()
    };
    let aux = aux_of(aux_data.clone());
    let base = fwd(&ids, &aux);
    assert_eq!(
        base.dims(),
        &[seq - 1, DRAFT_VOCAB],
        "an n-token row has n-1 shift rows: row i embeds token i+1, and the last token has no \
         aux-paired successor"
    );

    let mut ids0 = ids.clone();
    ids0[0] = (ids0[0] + 1) % VOCAB as u32;
    assert_eq!(
        max_abs_diff(&base, &fwd(&ids0, &aux)).unwrap(),
        0.0,
        "token 0 is never embedded under shift conditioning (serving's encode_context_rows feeds \
         embed(context[i+1]) to row i), so changing it must move no draft row; a nonzero diff \
         means the forward regressed to the unshifted embed(token i) pairing"
    );

    let p = 4usize;
    let mut idsp = ids.clone();
    idsp[p] = (idsp[p] + 1) % VOCAB as u32;
    let outp = fwd(&idsp, &aux);
    assert_eq!(
        max_abs_diff(
            &base.narrow(0, 0, p - 1).unwrap(),
            &outp.narrow(0, 0, p - 1).unwrap()
        )
        .unwrap(),
        0.0,
        "token {p} is embedded at shift row {}, so causal attention leaves every earlier row \
         bitwise unchanged",
        p - 1
    );
    assert!(
        max_abs_diff(
            &base.narrow(0, p - 1, 1).unwrap(),
            &outp.narrow(0, p - 1, 1).unwrap()
        )
        .unwrap()
            > 0.0,
        "changing token {p} must move shift row {} (the row that embeds it), or the embed index \
         is not i+1",
        p - 1
    );

    let mut aux_p = aux_data.clone();
    for v in aux_p[p * fc_in..(p + 1) * fc_in].iter_mut() {
        *v += 1.0;
    }
    let outa = fwd(&ids, &aux_of(aux_p));
    assert_eq!(
        max_abs_diff(
            &base.narrow(0, 0, p).unwrap(),
            &outa.narrow(0, 0, p).unwrap()
        )
        .unwrap(),
        0.0,
        "aux row {p} feeds shift row {p}, so rows before it stay bitwise unchanged"
    );
    assert!(
        max_abs_diff(
            &base.narrow(0, p, 1).unwrap(),
            &outa.narrow(0, p, 1).unwrap()
        )
        .unwrap()
            > 0.0,
        "perturbing aux row {p} must move shift row {p}, or the aux index is not i"
    );

    let mut aux_last = aux_data.clone();
    for v in aux_last[(seq - 1) * fc_in..].iter_mut() {
        *v += 1.0;
    }
    assert_eq!(
        max_abs_diff(&base, &fwd(&ids, &aux_of(aux_last))).unwrap(),
        0.0,
        "the final aux row pairs with a token beyond the row and must be unused in training"
    );
}

#[test]
fn the_distill_loss_matches_a_hand_computed_soft_ce_of_draft_row_i_against_teacher_row_i_plus_1() {
    let (_cfg, head) = fixture_head_f32();
    let seq = 4usize;
    let completion_start = 3usize;
    let mut rng = Lcg(0xd157_1077);
    let teacher_data: Vec<f32> = (0..seq * VOCAB).map(|_| rng.next_f32() * 3.0).collect();
    let draft_data: Vec<f32> = (0..(seq - 1) * DRAFT_VOCAB)
        .map(|_| rng.next_f32() * 3.0)
        .collect();
    let teacher =
        Tensor::from_vec(teacher_data.clone(), (seq, VOCAB), &Device::Cpu).unwrap();
    let draft =
        Tensor::from_vec(draft_data.clone(), (seq - 1, DRAFT_VOCAB), &Device::Cpu).unwrap();
    let got: f32 = head
        .shift_distill_loss_pairs_draft_row_i_with_teacher_row_i_plus_1(
            &draft,
            &teacher,
            completion_start,
        )
        .unwrap()
        .to_scalar()
        .unwrap();

    let fixture_d2t_offset_table_is_j_so_draft_j_maps_to_target_2j = |v: usize| 2 * v;
    let mut num = 0f64;
    let mut denom = 0f64;
    for i in 0..seq - 1 {
        let w = if i + 2 >= completion_start { 1.0f64 } else { 0.0 };
        denom += w;
        if w == 0.0 {
            continue;
        }
        let trow: Vec<f64> = (0..DRAFT_VOCAB)
            .map(|v| {
                teacher_data[(i + 1) * VOCAB
                    + fixture_d2t_offset_table_is_j_so_draft_j_maps_to_target_2j(v)]
                    as f64
            })
            .collect();
        let drow: Vec<f64> = draft_data[i * DRAFT_VOCAB..(i + 1) * DRAFT_VOCAB]
            .iter()
            .map(|x| *x as f64)
            .collect();
        let tmax = trow.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let tz: f64 = trow.iter().map(|x| (x - tmax).exp()).sum();
        let dmax = drow.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let dz: f64 = drow.iter().map(|x| (x - dmax).exp()).sum();
        for v in 0..DRAFT_VOCAB {
            let p = (trow[v] - tmax).exp() / tz;
            let logq = drow[v] - dmax - dz.ln();
            num += w * p * logq;
        }
    }
    let expected = -(num / denom);
    assert!(
        (got as f64 - expected).abs() < 1e-4,
        "distill loss must be the soft-CE of draft row i against the d2t-gathered teacher row \
         i+1, weighted by the prompt mask shifted to absolute predicted position i+2 (draft row \
         i drafts the token after token i+1, and the final row is serving's bonus draft): \
         hand-computed {expected:.6}, got {got}; any off-by-one in teacher row, mask position, \
         or d2t interpretation moves this value"
    );
}

fn expect_refusal(drafter: &Path, out_leaf: &str, wanted: &[&str]) {
    let f = fixture_dir();
    let mut args = train_args(&f.join(out_leaf));
    args.steps = 1;
    let mut spec = spec_with_rank_4();
    spec.drafter_dir = drafter.to_path_buf();
    let err = match run_with_speculator(&args, Some(&spec)) {
        Err(e) => format!("{e:#}"),
        Ok(_) => panic!(
            "drafter {} must be refused, but co-training ran",
            drafter.display()
        ),
    };
    for w in wanted {
        assert!(
            err.contains(w),
            "refusal for {} must name \"{w}\" so the caller learns what was expected; got: {err}",
            drafter.display()
        );
    }
}

#[test]
fn a_directory_without_a_drafter_config_is_refused_naming_the_expected_layout() {
    let d = fixture_dir().join("empty-drafter");
    std::fs::create_dir_all(&d).unwrap();
    expect_refusal(&d, "out-refuse-empty", &["config.json", "eagle3"]);
}

#[test]
fn a_non_eagle3_speculator_kind_is_refused_naming_both_kinds() {
    let d = fixture_dir().join("medusa-drafter");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(
        d.join("config.json"),
        drafter_config_json(HIDDEN, VOCAB, &AUX_IDS, "medusa"),
    )
    .unwrap();
    expect_refusal(&d, "out-refuse-kind", &["eagle3", "medusa"]);
}

#[test]
fn a_drafter_sized_for_another_target_is_refused_naming_the_hidden_sizes() {
    let d = fixture_dir().join("wide-drafter");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(
        d.join("config.json"),
        drafter_config_json(2 * HIDDEN, VOCAB, &AUX_IDS, "eagle3"),
    )
    .unwrap();
    expect_refusal(&d, "out-refuse-hidden", &["hidden_size", "64", "32"]);
}

#[test]
fn a_tap_beyond_the_built_layers_is_refused_naming_the_truncation_hazard() {
    let d = fixture_dir().join("deep-tap-drafter");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(
        d.join("config.json"),
        drafter_config_json(HIDDEN, VOCAB, &[0, 1, 7], "eagle3"),
    )
    .unwrap();
    expect_refusal(&d, "out-refuse-tap", &["taps layer 7", "NV_TRAIN_MAX_LAYERS"]);
}

#[test]
fn a_negative_tap_id_is_refused_naming_the_entry_instead_of_silently_dropping_it() {
    let d = fixture_dir().join("negative-tap-drafter");
    std::fs::create_dir_all(&d).unwrap();
    let cfg =
        drafter_config_json(HIDDEN, VOCAB, &AUX_IDS, "eagle3").replace("[0, 1, 2]", "[-1, 1, 2]");
    std::fs::write(d.join("config.json"), cfg).unwrap();
    expect_refusal(
        &d,
        "out-refuse-negtap",
        &["eagle_aux_hidden_state_layer_ids[0]", "-1", "fc_in_dim"],
    );
}

#[test]
fn a_config_without_weights_is_refused_naming_the_missing_safetensors() {
    let d = fixture_dir().join("weightless-drafter");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(
        d.join("config.json"),
        drafter_config_json(HIDDEN, VOCAB, &AUX_IDS, "eagle3"),
    )
    .unwrap();
    expect_refusal(&d, "out-refuse-weights", &["model.safetensors"]);
}
