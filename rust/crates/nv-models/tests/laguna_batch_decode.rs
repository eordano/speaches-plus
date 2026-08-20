#![cfg(feature = "cuda")]

mod common;
use common::laguna_chunked_prefill as chunked_prefill;
use common::argmax;
use common::distinct;
use common::envn;
use common::LcgTop24TwoSided as Lcg;
use common::prompt_for;
use common::rand_tensor;
use candle_core::{DType, Device, Tensor};
use nv_models::laguna::{Laguna, LagunaConfig, LagunaKvCache};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};
use common::laguna_xs21_nvfp4_snapshot_dir_env_override_then_home_hub;

const HIDDEN: usize = 64;
const INTER: usize = 128;
const MOE_INTER: usize = 32;
const SHARED_INTER: usize = 32;
const N_LAYERS: usize = 4;
const N_Q: usize = 4;
const N_KV: usize = 2;
const HEAD_DIM: usize = 16;
const VOCAB: usize = 512;
const N_EXPERTS: usize = 4;
const SLIDING_WINDOW: usize = 16;
const STEPS: usize = 8;
const SLOTS: usize = 4;
const REAL_B_GT_1_MAX_ABS_LOGIT_DRIFT_FROM_GROUPED_MOE_N_TOKENS_ARMS_MEASURED_2P2: f32 = 4.0;

fn one_gpu_test_at_a_time() -> MutexGuard<'static, ()> {
    static SERIALIZE: Mutex<()> = Mutex::new(());
    SERIALIZE.lock().unwrap_or_else(|e| e.into_inner())
}

fn solo_oracle_is_the_ungraphed_eager_path_captured_moe_equality_is_laguna_graph_gates_job() {
    std::env::set_var("NV_LAGUNA_GRAPH", "0");
}

fn near_one_tensor(rng: &mut Lcg, dim: usize) -> Tensor {
    let data: Vec<f32> = (0..dim).map(|_| 1.0 + rng.next_f32() * 0.1).collect();
    Tensor::from_vec(data, dim, &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn config_json() -> String {
    format!(
        r#"{{
  "architectures": ["LagunaForCausalLM"],
  "model_type": "laguna",
  "vocab_size": {VOCAB},
  "hidden_size": {HIDDEN},
  "intermediate_size": {INTER},
  "num_hidden_layers": {N_LAYERS},
  "num_attention_heads": {N_Q},
  "num_key_value_heads": {N_KV},
  "head_dim": {HEAD_DIM},
  "max_position_embeddings": 512,
  "rms_norm_eps": 1e-6,
  "num_experts": {N_EXPERTS},
  "num_experts_per_tok": 2,
  "moe_intermediate_size": {MOE_INTER},
  "shared_expert_intermediate_size": {SHARED_INTER},
  "norm_topk_prob": true,
  "tie_word_embeddings": true,
  "gating": "per-head",
  "sliding_window": {SLIDING_WINDOW},
  "moe_routed_scaling_factor": 1.5,
  "eos_token_id": [],
  "rope_parameters": {{
    "full_attention": {{
      "rope_theta": 500000.0,
      "rope_type": "yarn",
      "factor": 32.0,
      "original_max_position_embeddings": 64,
      "beta_slow": 1.0,
      "beta_fast": 64.0,
      "attention_factor": 1.3465735902799727,
      "partial_rotary_factor": 0.5
    }},
    "sliding_attention": {{
      "rope_type": "default",
      "rope_theta": 10000.0,
      "partial_rotary_factor": 1.0
    }}
  }},
  "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
  "mlp_layer_types": ["dense", "sparse", "sparse", "dense"],
  "num_attention_heads_per_layer": [{N_Q}, {N_Q}, {N_Q}, {N_Q}]
}}"#
    )
}

fn write_tiny_model(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();

    let mut rng = Lcg(0x1a9c_a5ee_d000_0001);
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert(
        "model.embed_tokens.weight".into(),
        rand_tensor(&mut rng, (VOCAB, HIDDEN), 1.0),
    );
    tensors.insert("model.norm.weight".into(), near_one_tensor(&mut rng, HIDDEN));
    for i in 0..N_LAYERS {
        let p = format!("model.layers.{i}");
        tensors.insert(
            format!("{p}.input_layernorm.weight"),
            near_one_tensor(&mut rng, HIDDEN),
        );
        tensors.insert(
            format!("{p}.post_attention_layernorm.weight"),
            near_one_tensor(&mut rng, HIDDEN),
        );
        tensors.insert(
            format!("{p}.self_attn.q_proj.weight"),
            rand_tensor(&mut rng, (N_Q * HEAD_DIM, HIDDEN), 0.3),
        );
        tensors.insert(
            format!("{p}.self_attn.k_proj.weight"),
            rand_tensor(&mut rng, (N_KV * HEAD_DIM, HIDDEN), 0.3),
        );
        tensors.insert(
            format!("{p}.self_attn.v_proj.weight"),
            rand_tensor(&mut rng, (N_KV * HEAD_DIM, HIDDEN), 0.3),
        );
        tensors.insert(
            format!("{p}.self_attn.o_proj.weight"),
            rand_tensor(&mut rng, (HIDDEN, N_Q * HEAD_DIM), 0.3),
        );
        tensors.insert(
            format!("{p}.self_attn.g_proj.weight"),
            rand_tensor(&mut rng, (N_Q, HIDDEN), 0.3),
        );
        tensors.insert(
            format!("{p}.self_attn.q_norm.weight"),
            near_one_tensor(&mut rng, HEAD_DIM),
        );
        tensors.insert(
            format!("{p}.self_attn.k_norm.weight"),
            near_one_tensor(&mut rng, HEAD_DIM),
        );
        let sparse = matches!(i, 1 | 2);
        if sparse {
            tensors.insert(
                format!("{p}.mlp.gate.weight"),
                rand_tensor(&mut rng, (N_EXPERTS, HIDDEN), 0.3),
            );
            for e in 0..N_EXPERTS {
                let ep = format!("{p}.mlp.experts.{e}");
                tensors.insert(
                    format!("{ep}.gate_proj.weight"),
                    rand_tensor(&mut rng, (MOE_INTER, HIDDEN), 0.3),
                );
                tensors.insert(
                    format!("{ep}.up_proj.weight"),
                    rand_tensor(&mut rng, (MOE_INTER, HIDDEN), 0.3),
                );
                tensors.insert(
                    format!("{ep}.down_proj.weight"),
                    rand_tensor(&mut rng, (HIDDEN, MOE_INTER), 0.3),
                );
            }
            let sp = format!("{p}.mlp.shared_expert");
            tensors.insert(
                format!("{sp}.gate_proj.weight"),
                rand_tensor(&mut rng, (SHARED_INTER, HIDDEN), 0.3),
            );
            tensors.insert(
                format!("{sp}.up_proj.weight"),
                rand_tensor(&mut rng, (SHARED_INTER, HIDDEN), 0.3),
            );
            tensors.insert(
                format!("{sp}.down_proj.weight"),
                rand_tensor(&mut rng, (HIDDEN, SHARED_INTER), 0.3),
            );
        } else {
            tensors.insert(
                format!("{p}.mlp.gate_proj.weight"),
                rand_tensor(&mut rng, (INTER, HIDDEN), 0.3),
            );
            tensors.insert(
                format!("{p}.mlp.up_proj.weight"),
                rand_tensor(&mut rng, (INTER, HIDDEN), 0.3),
            );
            tensors.insert(
                format!("{p}.mlp.down_proj.weight"),
                rand_tensor(&mut rng, (HIDDEN, INTER), 0.3),
            );
        }
    }
    candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
}

fn solo_decode(
    model: &Laguna,
    device: &Device,
    prompt: &[u32],
    steps: usize,
    max_seq: usize,
) -> (u32, Vec<u32>, Vec<Vec<u32>>) {
    let mut cache = model.new_kv_cache(max_seq).expect("solo cache");
    let last = chunked_prefill(model, &mut cache, prompt, device);
    let anchor = argmax(&last);
    let mut toks = Vec::with_capacity(steps);
    let mut bits = Vec::with_capacity(steps);
    let mut t = anchor;
    let mut pos = prompt.len();
    for _ in 0..steps {
        let tok = Tensor::from_vec(vec![t], (1usize, 1usize), device).unwrap();
        let p = Tensor::from_vec(vec![pos as i32], 1usize, device).unwrap();
        let logits = model
            .forward_with_cache(&tok, &p, &mut cache)
            .expect("solo decode step");
        let row: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        let n = argmax(&row);
        bits.push(row.into_iter().map(f32::to_bits).collect::<Vec<u32>>());
        toks.push(n);
        t = n;
        pos += 1;
    }
    (anchor, toks, bits)
}

struct StepDiff {
    diff_lanes: usize,
    max_abs: f32,
}

fn compare_rows(want_bits: &[u32], got: &[f32]) -> StepDiff {
    let mut diff_lanes = 0usize;
    let mut max_abs = 0f32;
    for (w, g) in want_bits.iter().zip(got.iter()) {
        if *w != g.to_bits() {
            diff_lanes += 1;
        }
        let d = (f32::from_bits(*w) - g).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    StepDiff {
        diff_lanes,
        max_abs,
    }
}

fn run_batched_against_solo(
    model: &Laguna,
    device: &Device,
    prompts: &[Vec<u32>],
    solo: &[(u32, Vec<u32>, Vec<Vec<u32>>)],
    steps: usize,
    max_seq: usize,
    label: &str,
    require_bit_identity: bool,
) {
    let b = prompts.len();
    let vocab = model.config().vocab_size;
    let mut lanes: Vec<LagunaKvCache> = (0..b)
        .map(|_| model.new_kv_cache(max_seq).expect("lane cache"))
        .collect();
    let mut cur: Vec<u32> = Vec::with_capacity(b);
    let mut pos: Vec<usize> = Vec::with_capacity(b);
    for (j, p) in prompts.iter().enumerate() {
        let last = chunked_prefill(model, &mut lanes[j], p, device);
        let anchor = argmax(&last);
        assert_eq!(
            anchor, solo[j].0,
            "{label}: lane {j} prefill anchor disagrees with the solo run before any batching"
        );
        cur.push(anchor);
        pos.push(p.len());
    }
    for i in 0..steps {
        let mut cache_refs: Vec<&mut LagunaKvCache> = lanes.iter_mut().collect();
        let logits = model
            .forward_decode_batched(&cur, &pos, &mut cache_refs)
            .expect("forward_decode_batched");
        let host: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(host.len(), b * vocab);
        for j in 0..b {
            let row = &host[j * vocab..(j + 1) * vocab];
            let d = compare_rows(&solo[j].2[i], row);
            if require_bit_identity {
                assert_eq!(
                    d.diff_lanes, 0,
                    "{label}: step {i} lane {j}: {} of {vocab} logit lanes differ from the same \
                     sequence run alone (max |delta| {:.3e})",
                    d.diff_lanes, d.max_abs
                );
            } else if d.diff_lanes > 0 {
                eprintln!(
                    "[lagbat] {label}: step {i} lane {j}: {}/{vocab} bit-diff lanes, max|delta| {:.3e}",
                    d.diff_lanes, d.max_abs
                );
                assert!(
                    d.max_abs < REAL_B_GT_1_MAX_ABS_LOGIT_DRIFT_FROM_GROUPED_MOE_N_TOKENS_ARMS_MEASURED_2P2,
                    "{label}: step {i} lane {j}: batched logit drift {:.3e} exceeds the recorded \
                     grouped-MoE arm-variance envelope; something beyond the known n_tokens>1 \
                     quantization-arm difference regressed",
                    d.max_abs
                );
            }
            let n = argmax(row);
            assert_eq!(
                n, solo[j].1[i],
                "{label}: step {i} lane {j}: sampled token differs from the solo run"
            );
            cur[j] = n;
            pos[j] += 1;
        }
    }
    eprintln!(
        "[lagbat] {label}: B={b}, {steps} steps token-identical to solo runs \
         (prompt lengths {:?}, bit_identity_required={require_bit_identity})",
        prompts.iter().map(|p| p.len()).collect::<Vec<_>>()
    );
}

#[test]
fn laguna_batch_decode_matches_single_stream() {
    if std::env::var("NV_LAGBAT_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_LAGBAT_TEST=1 to run this GPU gate, or NV_MODELS_ALLOW_SKIP=1 to skip \
                 it on purpose; a batched-decode gate must never silently report ok"
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_LAGBAT_TEST=1 to run");
        return;
    }
    let _gpu = one_gpu_test_at_a_time();
    solo_oracle_is_the_ungraphed_eager_path_captured_moe_equality_is_laguna_graph_gates_job();
    let device = Device::new_cuda(0).expect("cuda device 0");

    let dir = std::env::temp_dir().join(format!(
        "{}-{}",
        "nv-laguna-batch-decode-tiny",
        std::process::id()
    ));
    write_tiny_model(&dir);
    let cfg =
        LagunaConfig::from_hf_json_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
            .unwrap();
    let weights = WeightLoader::open_dir(&dir, &device).unwrap();
    let model =
        Laguna::from_loader_quantized(cfg, &weights, &QuantizationConfig::none(), &device).unwrap();
    drop(weights);

    let max_seq = 64usize;
    let prompts: Vec<Vec<u32>> = (0..SLOTS)
        .map(|j| prompt_for(j, 17 + 3 * j, VOCAB))
        .collect();
    let solo: Vec<(u32, Vec<u32>, Vec<Vec<u32>>)> = prompts
        .iter()
        .map(|p| solo_decode(&model, &device, p, STEPS, max_seq))
        .collect();

    let d = distinct(&solo[0].2[0]);
    assert!(
        d > VOCAB / 4,
        "logits are degenerate ({d} distinct of {VOCAB}); the compare would be vacuous"
    );
    for j in 1..SLOTS {
        let same = (0..STEPS).all(|i| solo[j].2[i] == solo[0].2[i]);
        assert!(
            !same,
            "lane {j}'s solo logits equal lane 0's at every step; the cross-lane compare would be vacuous"
        );
    }

    run_batched_against_solo(
        &model,
        &device,
        &prompts[..1],
        &solo[..1],
        STEPS,
        max_seq,
        "B=1",
        true,
    );
    run_batched_against_solo(
        &model,
        &device,
        &prompts,
        &solo,
        STEPS,
        max_seq,
        "B=4",
        true,
    );
    run_batched_against_solo(
        &model,
        &device,
        &prompts[..SLOTS - 1],
        &solo[..SLOTS - 1],
        STEPS,
        max_seq,
        "B=3",
        true,
    );
}

fn load_real_model(device: &Device) -> (Laguna, tokenizers::Tokenizer) {
    let dir = laguna_xs21_nvfp4_snapshot_dir_env_override_then_home_hub();
    eprintln!("[lagbat-real] loading {}", dir.display());
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = LagunaConfig::from_hf_json_str(&raw).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, device).expect("weights");
    let t = std::time::Instant::now();
    let model = Laguna::from_loader_quantized(cfg, &weights, &qcfg, device).expect("model");
    eprintln!("[lagbat-real] model built in {:.1}s", t.elapsed().as_secs_f64());
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    (model, tok)
}

fn real_prompts(tok: &tokenizers::Tokenizer, slots: usize) -> Vec<Vec<u32>> {
    let seeds = [
        "The measurement of record for this repository carries its basis with every number, and ",
        "A kernel is fast, a format is not; the only operative quantity is bytes over rate, and ",
        "Batching amortizes one weight read across more rows, which is the largest lever, and ",
        "Speculative decoding cannot pay on a compute-bound model at any acceptance rate, and ",
    ];
    (0..slots)
        .map(|j| {
            let mut text = String::new();
            let want = 48 + 7 * j;
            while tok.encode(text.as_str(), false).unwrap().get_ids().len() < want {
                text.push_str(seeds[j % seeds.len()]);
            }
            let mut ids: Vec<u32> = tok.encode(text.as_str(), false).unwrap().get_ids().to_vec();
            ids.truncate(want);
            ids
        })
        .collect()
}

#[test]
#[ignore = "loads the Laguna-XS-2.1 NVFP4 checkpoint; set NV_LAGBAT_REAL=1 -- batch-vs-solo parity on the real weights, solo oracle is the ungraphed eager path"]
fn real_laguna_xs21_batch_decode() {
    if std::env::var("NV_LAGBAT_REAL").as_deref() != Ok("1") {
        panic!("set NV_LAGBAT_REAL=1 to run this GPU test (it must never silently skip)");
    }
    let _gpu = one_gpu_test_at_a_time();
    solo_oracle_is_the_ungraphed_eager_path_captured_moe_equality_is_laguna_graph_gates_job();
    let device = Device::new_cuda(0).expect("cuda device 0");
    let (model, tok) = load_real_model(&device);
    let vocab = model.config().vocab_size;
    let max_seq = 512usize;
    let prompts = real_prompts(&tok, SLOTS);
    eprintln!(
        "[lagbat-real] prompt lengths {:?}",
        prompts.iter().map(|p| p.len()).collect::<Vec<_>>()
    );
    let solo: Vec<(u32, Vec<u32>, Vec<Vec<u32>>)> = prompts
        .iter()
        .map(|p| solo_decode(&model, &device, p, STEPS, max_seq))
        .collect();
    let d = distinct(&solo[0].2[0]);
    assert!(
        d > 1000,
        "logits are degenerate ({d} distinct of {vocab}); the compare would be vacuous"
    );
    for j in 1..SLOTS {
        assert!(
            (0..STEPS).any(|i| solo[j].2[i] != solo[0].2[i]),
            "lane {j}'s solo logits equal lane 0's at every step; the cross-lane compare would be vacuous"
        );
    }
    run_batched_against_solo(
        &model, &device, &prompts[..1], &solo[..1], STEPS, max_seq, "B=1", true,
    );
    run_batched_against_solo(&model, &device, &prompts, &solo, STEPS, max_seq, "B=4", false);
    run_batched_against_solo(
        &model,
        &device,
        &prompts[..SLOTS - 1],
        &solo[..SLOTS - 1],
        STEPS,
        max_seq,
        "B=3",
        false,
    );
}

fn prime_lane_to_depth(
    model: &Laguna,
    cache: &mut LagunaKvCache,
    depth: usize,
    device: &Device,
) -> u32 {
    let ids: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
    let last = chunked_prefill(model, cache, &ids, device);
    argmax(&last)
}

#[test]
#[ignore = "loads the Laguna-XS-2.1 NVFP4 checkpoint; set NV_LAGBAT_RATE=1 -- B-ladder B=1/2/4 at NV_LAGBAT_CTX depths (default 256,8192): per-lane + aggregate rate, plus the graphed-solo baseline so the eager batch path's graph gap is on record"]
fn real_laguna_xs21_batch_decode_rate() {
    if std::env::var("NV_LAGBAT_RATE").as_deref() != Ok("1") {
        panic!("set NV_LAGBAT_RATE=1 to run this GPU test (it must never silently skip)");
    }
    let _gpu = one_gpu_test_at_a_time();
    solo_oracle_is_the_ungraphed_eager_path_captured_moe_equality_is_laguna_graph_gates_job();
    let device = Device::new_cuda(0).expect("cuda device 0");
    let (model, _tok) = load_real_model(&device);
    let model = std::sync::Arc::new(model);
    let steps = envn("NV_LAGBAT_RATE_STEPS", 32);
    let reps = envn("NV_LAGBAT_RATE_REPS", 3);
    let depths: Vec<usize> = std::env::var("NV_LAGBAT_CTX")
        .unwrap_or_else(|_| "256,8192".to_string())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();

    for &depth in &depths {
        let max_seq = depth + steps * (reps + 2) + 64;

        {
            let mut g_cache = model.new_kv_cache(max_seq).expect("graph cache");
            let anchor = prime_lane_to_depth(&model, &mut g_cache, depth, &device);
            let mut sg = nv_models::laguna_step_graph::LagunaStepGraph::new(
                std::sync::Arc::clone(&model),
                g_cache,
            )
            .expect("step graph");
            let mut t0 = anchor;
            for _ in 0..steps {
                sg.step(t0).expect("graph warm step");
                t0 = sg.argmax_device().expect("graph argmax");
            }
            let mut ms = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t = std::time::Instant::now();
                for _ in 0..steps {
                    sg.step(t0).expect("graph step");
                    t0 = sg.argmax_device().expect("graph argmax");
                }
                ms.push(t.elapsed().as_secs_f64() * 1000.0 / steps as f64);
            }
            let mean = ms.iter().sum::<f64>() / ms.len() as f64;
            eprintln!(
                "[lagbat-rate] depth={depth} graphed-solo baseline {mean:.2} ms/tok ({:.1} tok/s)",
                1000.0 / mean
            );
        }

        for &b in &[1usize, 2, 4] {
            let mut lanes: Vec<LagunaKvCache> = (0..b)
                .map(|_| model.new_kv_cache(max_seq).expect("lane cache"))
                .collect();
            let mut cur: Vec<u32> = Vec::with_capacity(b);
            let mut pos: Vec<usize> = Vec::with_capacity(b);
            for lane in lanes.iter_mut() {
                cur.push(prime_lane_to_depth(&model, lane, depth, &device));
                pos.push(depth);
            }
            let step_once = |lanes: &mut Vec<LagunaKvCache>,
                             cur: &mut Vec<u32>,
                             pos: &mut Vec<usize>| {
                let mut refs: Vec<&mut LagunaKvCache> = lanes.iter_mut().collect();
                let logits = model
                    .forward_decode_batched(cur, pos, &mut refs)
                    .expect("batched step");
                let host: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
                let vocab = host.len() / cur.len();
                for j in 0..cur.len() {
                    cur[j] = argmax(&host[j * vocab..(j + 1) * vocab]);
                    pos[j] += 1;
                }
            };
            for _ in 0..steps {
                step_once(&mut lanes, &mut cur, &mut pos);
            }
            let mut ms = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t = std::time::Instant::now();
                for _ in 0..steps {
                    step_once(&mut lanes, &mut cur, &mut pos);
                }
                ms.push(t.elapsed().as_secs_f64() * 1000.0 / steps as f64);
            }
            let mean = ms.iter().sum::<f64>() / ms.len() as f64;
            eprintln!(
                "[lagbat-rate] depth={depth} B={b}: {mean:.2} ms/step per lane, {:.2} ms/tok \
                 aggregate, {:.1} tok/s aggregate",
                mean / b as f64,
                b as f64 * 1000.0 / mean
            );
        }
    }
}
