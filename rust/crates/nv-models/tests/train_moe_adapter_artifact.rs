use candle_core::{DType, Device, Tensor};
use nv_models::train_runner::{run, TrainArgs, TrainSummary};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const HIDDEN: usize = 64;
const INTER: usize = 96;
const MOE_INTER: usize = 24;
const N_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const N_LAYERS: usize = 3;
const N_Q: usize = 4;
const N_KV_SLIDING: usize = 2;
const N_KV_GLOBAL: usize = 1;
const HEAD_DIM_SLIDING: usize = 16;
const HEAD_DIM_GLOBAL: usize = 32;
const VOCAB: usize = 160;

const STEPS: usize = 3;
const LR: f64 = 0.05;
const RANK: usize = 4;

const SLIDING_LAYERS: [usize; 2] = [0, 1];
const THE_FULL_ATTENTION_LAYER_WHOSE_K_DOUBLES_AS_ITS_V: usize = 2;

const A_ZERO_LORA_B_AFTER_A_STEP_IS_A_MODULE_THAT_RECEIVED_NO_GRADIENT_AND_MERGES_AS_AN_EXACT_NO_OP: &str =
    "LoRA B is zero-initialised and only a gradient moves it, so a targeted module still exactly \
     zero after steps at a non-zero learning rate got no GradStore entry at all and merges into \
     the base as an exact no-op. The loss falls anyway, which is why no forward oracle catches \
     it. Two independent graph leaves put modules here, and q_proj, k_proj and v_proj are all \
     DEFAULT targets of nvk-train, train_runner::run and /v1/fine_tuning. The first is \
     gemma4_moe::naive_sdpa scoring through the fused candle_nn::ops::softmax_last_dim, whose \
     apply_op1_no_bwd output is a graph leaf: it strands q and k on the score path while v keeps \
     training through probs @ v, and alone it leaves q_proj dead in every layer and k_proj dead \
     only in the sliding layers, because on a base with attention_k_eq_v the full-attention \
     layer has no v_proj at all and its k_proj still trains through the v role it doubles as. \
     The second is Gemma4KvCache::write_at storing k and v with an in-place Tensor::slice_set, \
     so the view attention reads is Narrow of a buffer that is not on the graph: alone it leaves \
     every k_proj and every v_proj dead while q_proj trains. Read the dead set: q_proj implicates \
     the softmax, v_proj implicates the cache, and both together implicate both.";

const THE_ENDPOINT_GUARD_THIS_ROW_REPLICATES: &str =
    "/v1/fine_tuning reads the adapter back after a successful train_runner::run and fails the \
     job with code modules_received_no_gradient when a targeted module's lora_B is still exactly \
     zero, so a MoE fine-tune that trips this row is a MoE fine-tune the endpoint refuses to \
     hand a user. This row opens the same file with the same nv_weights::WeightLoader on the \
     same CPU device and takes the same amax over the same *.lora_B.weight names, so it passes \
     exactly when the endpoint would report success.";

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

fn ones(n: usize) -> Tensor {
    Tensor::from_vec(vec![1f32; n], n, &Device::Cpu).unwrap()
}

fn rand2(rng: &mut Lcg, rows: usize, cols: usize, scale: f32) -> Tensor {
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.next_f32() * scale).collect();
    Tensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap()
}

fn rand3(rng: &mut Lcg, a: usize, b: usize, c: usize, scale: f32) -> Tensor {
    let data: Vec<f32> = (0..a * b * c).map(|_| rng.next_f32() * scale).collect();
    Tensor::from_vec(data, (a, b, c), &Device::Cpu).unwrap()
}

fn config_json() -> String {
    format!(
        r#"{{
  "architectures": ["Gemma4ForCausalLM"],
  "model_type": "gemma4",
  "enable_moe_block": true,
  "num_experts": {N_EXPERTS},
  "top_k_experts": {TOP_K},
  "moe_intermediate_size": {MOE_INTER},
  "hidden_size": {HIDDEN},
  "intermediate_size": {INTER},
  "num_hidden_layers": {N_LAYERS},
  "num_attention_heads": {N_Q},
  "num_key_value_heads": {N_KV_SLIDING},
  "num_global_key_value_heads": {N_KV_GLOBAL},
  "head_dim": {HEAD_DIM_SLIDING},
  "global_head_dim": {HEAD_DIM_GLOBAL},
  "vocab_size": {VOCAB},
  "max_position_embeddings": 64,
  "rms_norm_eps": 1e-6,
  "sliding_window": 8,
  "final_logit_softcapping": 30.0,
  "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
  "attention_k_eq_v": true,
  "tie_word_embeddings": true,
  "hidden_activation": "gelu_pytorch_tanh",
  "rope_parameters": {{
    "full_attention": {{"rope_theta": 1000000.0, "partial_rotary_factor": 0.25}},
    "sliding_attention": {{"rope_theta": 10000.0}}
  }}
}}"#
    )
}

fn write_tiny_moe_model(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    let mut rng = Lcg(0x5eed_cafe_f00d_0198);
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "model.language_model.embed_tokens.weight".into(),
        rand2(&mut rng, VOCAB, HIDDEN, 1.0),
    );
    t.insert("model.language_model.norm.weight".into(), ones(HIDDEN));

    for i in 0..N_LAYERS {
        let p = format!("model.language_model.layers.{i}");
        let full = i == THE_FULL_ATTENTION_LAYER_WHOSE_K_DOUBLES_AS_ITS_V;
        let head_dim = if full {
            HEAD_DIM_GLOBAL
        } else {
            HEAD_DIM_SLIDING
        };
        let n_kv = if full { N_KV_GLOBAL } else { N_KV_SLIDING };
        let q_dim = N_Q * head_dim;
        let kv_dim = n_kv * head_dim;

        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "post_feedforward_layernorm_1",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_2",
        ] {
            t.insert(format!("{p}.{norm}.weight"), ones(HIDDEN));
        }
        t.insert(format!("{p}.layer_scalar"), ones(1));

        t.insert(
            format!("{p}.self_attn.q_proj.weight"),
            rand2(&mut rng, q_dim, HIDDEN, 0.3),
        );
        t.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rand2(&mut rng, kv_dim, HIDDEN, 0.3),
        );
        if !full {
            t.insert(
                format!("{p}.self_attn.v_proj.weight"),
                rand2(&mut rng, kv_dim, HIDDEN, 0.3),
            );
        }
        t.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rand2(&mut rng, HIDDEN, q_dim, 0.3),
        );
        t.insert(format!("{p}.self_attn.q_norm.weight"), ones(head_dim));
        t.insert(format!("{p}.self_attn.k_norm.weight"), ones(head_dim));

        t.insert(
            format!("{p}.mlp.gate_proj.weight"),
            rand2(&mut rng, INTER, HIDDEN, 0.3),
        );
        t.insert(
            format!("{p}.mlp.up_proj.weight"),
            rand2(&mut rng, INTER, HIDDEN, 0.3),
        );
        t.insert(
            format!("{p}.mlp.down_proj.weight"),
            rand2(&mut rng, HIDDEN, INTER, 0.3),
        );

        t.insert(
            format!("{p}.router.proj.weight"),
            rand2(&mut rng, N_EXPERTS, HIDDEN, 0.3),
        );
        t.insert(format!("{p}.router.scale"), ones(HIDDEN));
        t.insert(format!("{p}.router.per_expert_scale"), ones(N_EXPERTS));
        t.insert(
            format!("{p}.experts.gate_up_proj"),
            rand3(&mut rng, N_EXPERTS, 2 * MOE_INTER, HIDDEN, 0.3),
        );
        t.insert(
            format!("{p}.experts.down_proj"),
            rand3(&mut rng, N_EXPERTS, HIDDEN, MOE_INTER, 0.3),
        );
    }
    candle_core::safetensors::save(&t, dir.join("model.safetensors")).unwrap();
}

struct TrainedMoeAdapter {
    out: PathBuf,
    summary: TrainSummary,
}

static TRAINED: OnceLock<TrainedMoeAdapter> = OnceLock::new();

fn trained_moe_adapter() -> &'static TrainedMoeAdapter {
    TRAINED.get_or_init(|| {
        std::env::remove_var("NV_TRAIN_CKPT");
        let dir =
            std::env::temp_dir().join(format!("nv-train-moe-artifact-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let base = dir.join("base");
        write_tiny_moe_model(&base);

        let mut rng = Lcg(0xc0de_0198);
        let mut jsonl = String::new();
        for _ in 0..4 {
            let ids: Vec<u32> = (0..16).map(|_| rng.token()).collect();
            jsonl.push_str(&format!("{{\"ids\":{ids:?}}}\n"));
        }
        let data = dir.join("moe.jsonl");
        std::fs::write(&data, jsonl).unwrap();
        let out = dir.join("out-moe");

        let summary = run(&TrainArgs {
            base,
            data,
            eval_data: None,
            out: out.clone(),
            rank: RANK,
            alpha: 8.0,
            targets: TrainArgs::default_targets(),
            steps: STEPS,
            lr: LR,
            seed: 11,
        })
        .unwrap();
        eprintln!("[moe] losses {:?}", summary.losses);
        eprintln!("[moe] modules {:?}", summary.modules);
        TrainedMoeAdapter { out, summary }
    })
}

fn lora_b_amax_by_module_through_the_endpoints_own_reader(
    adapter_dir: &Path,
) -> Vec<(String, f32)> {
    let path = adapter_dir.join("adapter_model.safetensors");
    let weights = nv_weights::WeightLoader::open_file(&path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut out = Vec::new();
    for name in weights.names() {
        if !name.ends_with("lora_B.weight") {
            continue;
        }
        let amax = weights
            .get(&name, DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        out.push((
            name.trim_start_matches("base_model.model.")
                .trim_end_matches(".lora_B.weight")
                .to_string(),
            amax,
        ));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn modules_that_received_no_gradient(rows: &[(String, f32)]) -> Vec<&str> {
    rows.iter()
        .filter(|(_, amax)| *amax == 0.0)
        .map(|(m, _)| m.as_str())
        .collect()
}

fn assert_the_fixture_is_the_moe_k_eq_v_shape_the_damaged_adapter_had(
    trained: &TrainedMoeAdapter,
    rows: &[(String, f32)],
) {
    assert!(
        !trained.summary.dense_base,
        "this row loaded a DENSE base, so it re-tests dense_train::naive_sdpa and proves nothing \
         about gemma4_moe::naive_sdpa"
    );
    assert_eq!(
        trained.summary.losses.len(),
        STEPS,
        "the trainer did not take the steps this row's zero-check is stated against"
    );
    let names: Vec<&str> = rows.iter().map(|(m, _)| m.as_str()).collect();
    for li in SLIDING_LAYERS {
        for want in ["q_proj", "k_proj", "v_proj"] {
            let module = format!("model.language_model.layers.{li}.self_attn.{want}");
            assert!(
                names.contains(&module.as_str()),
                "the adapter carries no {module}, so this row would pass by containing nothing"
            );
        }
    }
    let full = THE_FULL_ATTENTION_LAYER_WHOSE_K_DOUBLES_AS_ITS_V;
    for want in ["q_proj", "k_proj"] {
        let module = format!("model.language_model.layers.{full}.self_attn.{want}");
        assert!(
            names.contains(&module.as_str()),
            "the adapter carries no {module}, so the full-attention layer is not under test"
        );
    }
    let v_on_full = format!("model.language_model.layers.{full}.self_attn.v_proj");
    assert!(
        !names.contains(&v_on_full.as_str()),
        "layer {full} emitted a v_proj, so this fixture is not the attention_k_eq_v shape the \
         damaged adapter was trained on and the k-trains-through-its-v-role case goes untested"
    );
    assert_eq!(
        rows.len(),
        3 * 6 + 2,
        "the adapter holds {} lora_B entries, not the 20 a 3-layer attention_k_eq_v MoE base \
         emits for the default target list",
        rows.len()
    );
}

#[test]
fn every_default_lora_target_on_a_moe_base_moves_off_its_zero_init_b_including_q_proj_and_k_proj() {
    let trained = trained_moe_adapter();
    let rows = lora_b_amax_by_module_through_the_endpoints_own_reader(&trained.out);
    for (module, amax) in &rows {
        eprintln!("[moe] {module} lora_B amax {amax:e}");
    }
    assert_the_fixture_is_the_moe_k_eq_v_shape_the_damaged_adapter_had(trained, &rows);
    let dead = modules_that_received_no_gradient(&rows);
    assert!(
        dead.is_empty(),
        "{} of {} targeted modules are still exactly zero after {STEPS} steps at lr {LR} \
         ({dead:?}), while the loss went {:?}. \
         {A_ZERO_LORA_B_AFTER_A_STEP_IS_A_MODULE_THAT_RECEIVED_NO_GRADIENT_AND_MERGES_AS_AN_EXACT_NO_OP}",
        dead.len(),
        rows.len(),
        trained.summary.losses
    );
}

#[test]
fn the_fine_tuning_endpoint_guard_finds_no_module_without_a_gradient_on_a_moe_fine_tune() {
    let trained = trained_moe_adapter();
    let rows = lora_b_amax_by_module_through_the_endpoints_own_reader(&trained.out);
    let dead = modules_that_received_no_gradient(&rows);
    eprintln!(
        "[moe] endpoint guard: {} of {} lora_B still exactly zero",
        dead.len(),
        rows.len()
    );
    assert!(
        !rows.is_empty(),
        "the guard read no lora_B tensors at all from {}, so it would pass on any adapter",
        trained.out.display()
    );
    assert!(
        dead.is_empty(),
        "a MoE fine-tune through train_runner::run still produces {} untrained modules \
         ({dead:?}), so /v1/fine_tuning fails this job with code modules_received_no_gradient \
         instead of completing it. {THE_ENDPOINT_GUARD_THIS_ROW_REPLICATES}",
        dead.len()
    );
}
