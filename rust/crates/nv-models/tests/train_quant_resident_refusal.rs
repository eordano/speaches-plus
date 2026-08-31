use candle_core::{DType, Device, Tensor};
use nv_models::train_runner::{run, TrainArgs};
use nv_train::qwen38::NV_TRAIN_QUANT_RESIDENT_KEEPS_NVFP4_AND_FP8_MODULES_RESIDENT_SO_TRAINING_RUNS_THE_SERVING_QUANT_GEMM as ARM;
use std::collections::HashMap;
use std::path::Path;

const HIDDEN: usize = 32;
const INTER: usize = 64;
const N_LAYERS: usize = 2;
const N_Q: usize = 2;
const N_KV: usize = 1;
const HEAD_DIM: usize = 16;
const VOCAB: usize = 128;

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as u32 as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
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

fn write_tiny_dense_bf16_model(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let config = format!(
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
    );
    std::fs::write(dir.join("config.json"), config).unwrap();
    let mut rng = Lcg(0x5eed_0087);
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

#[test]
fn the_gemma_dense_arm_refuses_the_resident_env_on_an_unpacked_bf16_base() {
    let root = std::env::temp_dir().join(format!("nv-models-qres-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let base = root.join("base");
    write_tiny_dense_bf16_model(&base);
    let data = root.join("train.jsonl");
    std::fs::write(&data, "{\"text\": \"tiny row for the refusal path\"}\n").unwrap();
    std::env::set_var(ARM, "1");
    let err = run(&TrainArgs {
        base,
        data,
        eval_data: None,
        out: root.join("adapter"),
        rank: 2,
        alpha: 2.0,
        targets: TrainArgs::default_targets(),
        steps: 1,
        lr: 1e-3,
        seed: 3,
    });
    std::env::remove_var(ARM);
    let msg = format!("{:#}", err.expect_err("a bf16 dense base must refuse the resident arm"));
    assert!(
        msg.contains("NV_TRAIN_QUANT_RESIDENT") && msg.contains("nvfp4-packed"),
        "the refusal must name the arm and the missing storage, got: {msg}"
    );
    assert!(
        !root.join("adapter").join("adapter_model.safetensors").exists(),
        "a refused arm must not leave an adapter behind"
    );
    let _ = std::fs::remove_dir_all(&root);
}
