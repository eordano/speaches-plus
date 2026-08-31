mod common;
use candle_core::{DType, Device, Tensor};
use common::ones_tensor;
use nv_models::train_runner::{load_base, run, BaseModel, TrainArgs};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const HIDDEN: usize = 128;
const INTER: usize = 256;
const N_LAYERS: usize = 2;
const N_Q: usize = 2;
const N_KV: usize = 1;
const HEAD_DIM: usize = 128;
const VOCAB: usize = 512;

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

fn config_json() -> String {
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
  "layer_types": ["full_attention", "sliding_attention"],
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

fn write_tiny_model(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    let mut rng = Lcg(0x5eed_cafe_f00d_0087);
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "model.language_model.embed_tokens.weight".into(),
        rand_tensor(&mut rng, (VOCAB, HIDDEN), 1.0),
    );
    t.insert(
        "model.language_model.norm.weight".into(),
        ones_tensor(HIDDEN),
    );
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
            t.insert(format!("{p}.{norm}.weight"), ones_tensor(HIDDEN));
        }
        t.insert(format!("{p}.layer_scalar"), ones_tensor(1));
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
        t.insert(
            format!("{p}.self_attn.q_norm.weight"),
            ones_tensor(HEAD_DIM),
        );
        t.insert(
            format!("{p}.self_attn.k_norm.weight"),
            ones_tensor(HEAD_DIM),
        );
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

static FIXTURE: OnceLock<PathBuf> = OnceLock::new();

fn fixture_dir() -> &'static Path {
    FIXTURE.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("nv-train-batchfwd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let base = dir.join("base");
        write_tiny_model(&base);
        dir
    })
}

fn rows(n: usize, seq: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| (0..seq).map(|_| rng.token()).collect())
        .collect()
}

fn solo_logits(model: &BaseModel, ids: &[u32], dev: &Device) -> Tensor {
    let seq = ids.len();
    let tokens = Tensor::from_vec(ids.to_vec(), (1usize, seq), dev).unwrap();
    let positions = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, dev).unwrap();
    model.forward_logits(&tokens, &positions).unwrap()
}

fn stacked_vs_solo_worst_rel(dev: &Device) -> f64 {
    let base = fixture_dir().join("base");
    let (model, _) = load_base(&base, dev).unwrap();
    let seq = 24usize;
    let batch = rows(3, seq, 0xbeef_0087);

    let flat: Vec<u32> = batch.iter().flatten().copied().collect();
    let tokens = Tensor::from_vec(flat, (3usize, seq), dev).unwrap();
    let positions = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, dev).unwrap();
    let stacked = model.forward_logits(&tokens, &positions).unwrap();
    assert_eq!(
        stacked.dims(),
        &[3, seq, VOCAB],
        "a batch of 3 must come back as 3 rows of logits"
    );

    let mut worst = 0f64;
    for (bi, ids) in batch.iter().enumerate() {
        let alone: Vec<f32> = solo_logits(&model, ids, dev)
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let row: Vec<f32> = stacked
            .narrow(0, bi, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(alone.len(), row.len());
        for (a, s) in alone.iter().zip(row.iter()) {
            assert!(
                s.is_finite(),
                "row {bi}: stacked forward produced non-finite logit"
            );
            let rel = (*a as f64 - *s as f64).abs() / (a.abs() as f64).max(1.0);
            worst = worst.max(rel);
        }
    }
    worst
}

#[test]
fn each_stacked_row_gets_the_logits_it_would_have_had_alone_cpu() {
    let worst = stacked_vs_solo_worst_rel(&Device::Cpu);
    eprintln!("[batchfwd cpu] worst rel {worst:.3e}");
    assert!(
        worst <= 1e-5,
        "cpu f32: a stacked row differs from its solo forward by {worst:.3e} relative; \
         the batch port changed the model, not just its speed"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs cuda; run with --ignored"]
fn each_stacked_row_gets_the_logits_it_would_have_had_alone_cuda_bf16() {
    let dev = Device::new_cuda(0).expect("cuda device 0");
    let worst = stacked_vs_solo_worst_rel(&dev);
    eprintln!("[batchfwd cuda] worst rel {worst:.3e}");
    assert!(
        worst <= 1e-3,
        "cuda bf16: a stacked row differs from its solo forward by {worst:.3e} relative \
         (measured noise on this board is 6e-6), so the batch port changed the model"
    );
}

fn ce_mean_of_row(model: &BaseModel, ids: &[u32], dev: &Device) -> f32 {
    let seq = ids.len();
    let logits = solo_logits(model, ids, dev);
    let inp = logits
        .squeeze(0)
        .unwrap()
        .narrow(0, 0, seq - 1)
        .unwrap()
        .contiguous()
        .unwrap();
    let tgt = Tensor::from_vec(ids[1..].to_vec(), seq - 1, dev).unwrap();
    candle_nn::loss::cross_entropy(&inp, &tgt)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

static NV_TRAIN_CKPT_IS_PROCESS_WIDE_SO_EVERY_RUN_CALLER_IN_THIS_BINARY_TAKES_THIS: OnceLock<
    std::sync::Mutex<()>,
> = OnceLock::new();

fn one_run_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    NV_TRAIN_CKPT_IS_PROCESS_WIDE_SO_EVERY_RUN_CALLER_IN_THIS_BINARY_TAKES_THIS
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

const A_ZERO_LORA_B_AFTER_A_STEP_IS_A_MODULE_THAT_RECEIVED_NO_GRADIENT_AND_MERGES_AS_AN_EXACT_NO_OP: &str =
    "LoRA B is zero-initialised and only a gradient moves it, so a targeted module still exactly \
     zero after steps at a non-zero learning rate got no GradStore entry at all and merges into \
     the base as an exact no-op. The loss falls anyway, which is why no other row here catches \
     it. The cause this row exists for is dense_train::naive_sdpa scoring through the fused \
     candle_nn::ops::softmax_last_dim, whose apply_op1_no_bwd output is a graph leaf: q and k \
     become unreachable from the loss while v keeps training through probs @ v, and q_proj and \
     k_proj are DEFAULT targets of nvk-train, train_runner::run and /v1/fine_tuning.";

fn lora_b_amax_by_module(adapter_dir: &Path) -> Vec<(String, f32)> {
    let path = adapter_dir.join("adapter_model.safetensors");
    let tensors = candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut out: Vec<(String, f32)> = tensors
        .iter()
        .filter(|(name, _)| name.ends_with(".lora_B.weight"))
        .map(|(name, t)| {
            let amax = t
                .to_dtype(DType::F32)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            (
                name.trim_start_matches("base_model.model.")
                    .trim_end_matches(".lora_B.weight")
                    .to_string(),
                amax,
            )
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn every_default_target_moves_off_zero(checkpointed: bool) {
    let _guard = one_run_at_a_time();
    if checkpointed {
        std::env::set_var("NV_TRAIN_CKPT", "1");
    } else {
        std::env::remove_var("NV_TRAIN_CKPT");
    }
    let dir = fixture_dir();
    let base = dir.join("base");
    let examples = rows(4, 16, 0xc0de_0087);
    let mut jsonl = String::new();
    for ids in &examples {
        jsonl.push_str(&format!("{{\"ids\":{ids:?}}}\n"));
    }
    let tag = if checkpointed { "ckpt" } else { "plain" };
    let data = dir.join(format!("grad-{tag}.jsonl"));
    std::fs::write(&data, jsonl).unwrap();
    let out = dir.join(format!("out-grad-{tag}"));

    let summary = run(&TrainArgs {
        base,
        data,
        eval_data: None,
        out: out.clone(),
        rank: 4,
        alpha: 8.0,
        targets: TrainArgs::default_targets(),
        steps: 3,
        lr: 0.05,
        seed: 11,
    })
    .unwrap();
    std::env::remove_var("NV_TRAIN_CKPT");

    assert_eq!(
        summary.checkpointed, checkpointed,
        "the {tag} arm did not take the code path it names"
    );
    let rows = lora_b_amax_by_module(&out);
    eprintln!("[{tag}] losses {:?}", summary.losses);
    for (module, amax) in &rows {
        eprintln!("[{tag}] {module} lora_B amax {amax:e}");
    }
    let names: Vec<&str> = rows.iter().map(|(m, _)| m.as_str()).collect();
    for want in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] {
        assert!(
            names.iter().filter(|n| n.ends_with(want)).count() == N_LAYERS,
            "{tag}: the adapter carries {} {want} entries, not {N_LAYERS} -- this row proves \
             nothing unless every layer's projection is actually in it",
            names.iter().filter(|n| n.ends_with(want)).count()
        );
    }
    let dead: Vec<&str> = rows
        .iter()
        .filter(|(_, amax)| *amax == 0.0)
        .map(|(m, _)| m.as_str())
        .collect();
    assert!(
        dead.is_empty(),
        "{tag}: {} of {} targeted modules are still exactly zero after 3 steps at lr 0.05 \
         ({dead:?}). {A_ZERO_LORA_B_AFTER_A_STEP_IS_A_MODULE_THAT_RECEIVED_NO_GRADIENT_AND_MERGES_AS_AN_EXACT_NO_OP}",
        dead.len(),
        rows.len()
    );
}

#[test]
fn every_default_lora_target_moves_off_its_zero_init_b_including_q_proj_and_k_proj() {
    every_default_target_moves_off_zero(false);
}

#[test]
fn the_checkpointed_dense_step_also_moves_q_proj_and_k_proj_off_zero() {
    every_default_target_moves_off_zero(true);
}

fn masked_ce_mean_of_row(model: &BaseModel, ids: &[u32], completion_start: usize, dev: &Device) -> f32 {
    let seq = ids.len();
    let logits = solo_logits(model, ids, dev);
    let logp = candle_nn::ops::log_softmax(
        &logits
            .squeeze(0)
            .unwrap()
            .narrow(0, 0, seq - 1)
            .unwrap()
            .contiguous()
            .unwrap(),
        candle_core::D::Minus1,
    )
    .unwrap();
    let host: Vec<f32> = logp.flatten_all().unwrap().to_vec1().unwrap();
    let mut total = 0f64;
    let mut count = 0usize;
    for j in 0..seq - 1 {
        if j + 1 >= completion_start {
            total += host[j * VOCAB + ids[j + 1] as usize] as f64;
            count += 1;
        }
    }
    (-total / count as f64) as f32
}

#[test]
fn prompt_positions_are_scored_by_neither_the_plain_nor_the_checkpointed_dense_step() {
    let _guard = one_run_at_a_time();
    let dir = fixture_dir();
    let base = dir.join("base");
    let dev = Device::Cpu;

    let prompt = "what did critics praise?";
    let completion = " the art direction and the score";
    let data = dir.join("masked.jsonl");
    std::fs::write(
        &data,
        format!(
            "{}\n",
            serde_json::json!({"prompt": prompt, "completion": completion})
        ),
    )
    .unwrap();

    let ids: Vec<u32> = format!("{prompt}{completion}")
        .bytes()
        .map(|b| (b as u32) % VOCAB as u32)
        .collect();
    let (model, _) = load_base(&base, &dev).unwrap();
    let expected = masked_ce_mean_of_row(&model, &ids, prompt.len(), &dev);
    let unmasked = ce_mean_of_row(&model, &ids, &dev);
    assert!(
        (expected - unmasked).abs() > 1e-4,
        "masked ({expected}) and all-position ({unmasked}) CE coincide on this row, so this \
         test cannot tell whether the mask is applied at all"
    );

    for (tag, ckpt) in [("plain", "0"), ("ckpt", "1")] {
        std::env::set_var("NV_TRAIN_CKPT", ckpt);
        let summary = run(&TrainArgs {
            base: base.clone(),
            data: data.clone(),
            eval_data: None,
            out: dir.join(format!("out-masked-{tag}")),
            rank: 4,
            alpha: 8.0,
            targets: vec!["q".into(), "v".into()],
            steps: 1,
            lr: 0.0,
            seed: 7,
        })
        .unwrap();
        std::env::remove_var("NV_TRAIN_CKPT");
        assert_eq!(summary.checkpointed, tag == "ckpt");
        let got = summary.losses[0];
        let rel = ((got - expected) / expected.abs().max(1e-6)).abs();
        assert!(
            rel <= 1e-4,
            "{tag}: step-0 loss {got:.6e} != completion-only CE mean {expected:.6e} \
             (rel {rel:.3e}); prompt positions are leaking into the training loss"
        );
    }
}

#[test]
fn step_loss_still_sums_one_ce_mean_per_example_across_mixed_length_buckets() {
    let _guard = one_run_at_a_time();
    std::env::remove_var("NV_TRAIN_CKPT");
    let dir = fixture_dir();
    let base = dir.join("base");
    let dev = Device::Cpu;

    let mut examples = rows(3, 12, 0xaaaa_0087);
    examples.extend(rows(2, 20, 0xbbbb_0087));
    let mut jsonl = String::new();
    for ids in &examples {
        jsonl.push_str(&format!("{{\"ids\":{ids:?}}}\n"));
    }
    let data = dir.join("mixed.jsonl");
    std::fs::write(&data, jsonl).unwrap();

    let (model, _) = load_base(&base, &dev).unwrap();
    let expected: f32 = examples
        .iter()
        .map(|ids| ce_mean_of_row(&model, ids, &dev))
        .sum();

    let summary = run(&TrainArgs {
        base: base.clone(),
        data,
        eval_data: None,
        out: dir.join("out-mixed"),
        rank: 4,
        alpha: 8.0,
        targets: vec!["q".into(), "v".into()],
        steps: 1,
        lr: 0.0,
        seed: 7,
    })
    .unwrap();
    assert_eq!(summary.losses.len(), 1);
    let got = summary.losses[0];
    let rel = ((got - expected) / expected.abs().max(1e-6)).abs();
    eprintln!("[bucket loss] expected {expected:.6e} got {got:.6e} rel {rel:.3e}");
    assert!(
        rel <= 1e-4,
        "step-0 loss {got:.6e} != sum of per-example CE means {expected:.6e} \
         (rel {rel:.3e}); LoRA B starts at zero so the forwards are identical, meaning \
         the bucketed loss dropped or double-counted an example, most likely by \
         forgetting that cross_entropy returns a MEAN and each bucket must be \
         rescaled by its example count"
    );
}

#[test]
fn heldout_eval_emits_per_row_paired_ce_in_dataset_order_across_buckets_and_solo() {
    let _guard = one_run_at_a_time();
    std::env::remove_var("NV_TRAIN_CKPT");
    let dir = fixture_dir();
    let base = dir.join("base");
    let dev = Device::Cpu;

    let train = rows(2, 12, 0x1111_0087);
    let mut jsonl = String::new();
    for ids in &train {
        jsonl.push_str(&format!("{{\"ids\":{ids:?}}}\n"));
    }
    let data = dir.join("evalvec-train.jsonl");
    std::fs::write(&data, jsonl).unwrap();

    let lens = [16usize, 8, 16, 12];
    let eval_ids: Vec<Vec<u32>> = lens
        .iter()
        .enumerate()
        .map(|(i, l)| rows(1, *l, 0x2222_0087 + i as u64).remove(0))
        .collect();
    let mut ejsonl = String::new();
    for ids in &eval_ids {
        ejsonl.push_str(&format!("{{\"ids\":{ids:?}}}\n"));
    }
    let eval = dir.join("evalvec-eval.jsonl");
    std::fs::write(&eval, ejsonl).unwrap();

    let args = |out: &str| TrainArgs {
        base: base.clone(),
        data: data.clone(),
        eval_data: Some(eval.clone()),
        out: dir.join(out),
        rank: 4,
        alpha: 8.0,
        targets: vec!["q".into(), "v".into()],
        steps: 0,
        lr: 0.0,
        seed: 7,
    };
    let bucketed = run(&args("out-evalvec-bucketed")).unwrap();
    assert_eq!(bucketed.num_eval_examples, 4);
    assert_eq!(bucketed.eval_ce_base.len(), 4);
    assert_eq!(
        bucketed.eval_ce_adapter, bucketed.eval_ce_base,
        "a zero-B adapter is an exact no-op, so every paired per-row delta is exactly zero"
    );
    assert_eq!(
        Some(bucketed.eval_ce_base.iter().sum::<f32>()),
        bucketed.eval_loss_base,
        "eval_loss_base must stay exactly the sum of the per-row base CE means"
    );
    assert_eq!(
        Some(bucketed.eval_ce_adapter.iter().sum::<f32>()),
        bucketed.eval_loss_adapter,
        "eval_loss_adapter must stay exactly the sum of the per-row adapter CE means"
    );

    let (model, _) = load_base(&base, &dev).unwrap();
    for (i, ids) in eval_ids.iter().enumerate() {
        let want = ce_mean_of_row(&model, ids, &dev);
        let got = bucketed.eval_ce_base[i];
        let rel = ((got - want) / want.abs().max(1e-6)).abs();
        assert!(
            rel <= 1e-4,
            "row {i}: eval_ce_base {got:.6e} != its solo-forward CE mean {want:.6e} \
             (rel {rel:.3e}); the bucketed eval returned rows out of dataset order"
        );
    }

    std::env::set_var("NV_TRAIN_SOLO", "1");
    let solo = run(&args("out-evalvec-solo")).unwrap();
    std::env::remove_var("NV_TRAIN_SOLO");
    for (i, (b, s)) in bucketed
        .eval_ce_base
        .iter()
        .zip(&solo.eval_ce_base)
        .enumerate()
    {
        let rel = ((b - s) / s.abs().max(1e-6)).abs();
        assert!(
            rel <= 1e-4,
            "row {i}: bucketed {b:.6e} vs solo {s:.6e} (rel {rel:.3e}); the two eval arms \
             disagree on the same row, so they are not computing one statistic"
        );
    }
}

#[test]
fn eval_ce_honors_the_prompt_boundary_row_for_row_on_identical_token_streams() {
    let _guard = one_run_at_a_time();
    std::env::remove_var("NV_TRAIN_CKPT");
    let dir = fixture_dir();
    let base = dir.join("base");
    let prompt = "The Valkyria series began ";
    let completion = "as a tactical role playing franchise.";
    let whole = format!("{prompt}{completion}");
    let data = dir.join("evalmask-train.jsonl");
    std::fs::write(
        &data,
        format!("{}\n", serde_json::json!({ "text": whole.as_str() })),
    )
    .unwrap();
    let eval = dir.join("evalmask-eval.jsonl");
    std::fs::write(
        &eval,
        format!(
            "{}\n{}\n{}\n",
            serde_json::json!({ "text": whole.as_str() }),
            serde_json::json!({ "prompt": "", "completion": whole.as_str() }),
            serde_json::json!({ "prompt": prompt, "completion": completion }),
        ),
    )
    .unwrap();
    let summary = run(&TrainArgs {
        base,
        data,
        eval_data: Some(eval),
        out: dir.join("out-evalmask"),
        rank: 4,
        alpha: 8.0,
        targets: vec!["q".into(), "v".into()],
        steps: 0,
        lr: 0.0,
        seed: 7,
    })
    .unwrap();
    assert_eq!(summary.num_eval_examples, 3);
    let ce = &summary.eval_ce_base;
    assert_eq!(
        ce[0], ce[1],
        "byte tokenization makes rows 0 and 1 the same ids with the same all-positions mask, so \
         their per-row CE must match bitwise"
    );
    assert_ne!(
        ce[2], ce[0],
        "row 2 carries the same ids but a completion-only mask; equal CE would mean the prompt \
         boundary is ignored and eval scores prompt positions"
    );
}
