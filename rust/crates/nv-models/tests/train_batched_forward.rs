mod common;
use common::ones_tensor;
use candle_core::{DType, Device, Tensor};
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
        let dir =
            std::env::temp_dir().join(format!("nv-train-batchfwd-{}", std::process::id()));
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
            assert!(s.is_finite(), "row {bi}: stacked forward produced non-finite logit");
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

#[test]
fn step_loss_still_sums_one_ce_mean_per_example_across_mixed_length_buckets() {
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
