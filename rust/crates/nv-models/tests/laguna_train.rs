use candle_core::{DType, Device, Tensor, Var};
use nv_models::laguna::{Laguna, LagunaConfig};
use nv_models::laguna_train::{
    run, tiny_laguna_config_json, write_tiny_laguna_checkpoint, LagunaTrainArgs, LagunaTrainModel,
};
use nv_weights::WeightLoader;
use std::path::{Path, PathBuf};

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nv-laguna-train-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_jsonl(path: &Path, rows: &[&str]) {
    let mut body = String::new();
    for r in rows {
        body.push_str(&serde_json::json!({ "text": r }).to_string());
        body.push('\n');
    }
    std::fs::write(path, body).unwrap();
}

fn args_for(root: &Path) -> LagunaTrainArgs {
    let base = root.join("base");
    write_tiny_laguna_checkpoint(&base);
    let data = root.join("train.jsonl");
    write_jsonl(
        &data,
        &[
            "The battle system alternates a command map with an action phase.",
            "Critics praised the art direction and the score of the third entry.",
        ],
    );
    let eval = root.join("eval.jsonl");
    write_jsonl(
        &eval,
        &[
            "The remake keeps the overworld theme and rescores the finale.",
            "A hidden dungeon opens after the credits roll.",
        ],
    );
    LagunaTrainArgs {
        base,
        data,
        eval_data: Some(eval),
        out: root.join("adapter"),
        rank: 4,
        alpha: 4.0,
        targets: LagunaTrainArgs::default_targets(),
        steps: 4,
        lr: 1e-2,
        seed: 3,
    }
}

fn amax(t: &Tensor) -> f32 {
    t.to_dtype(DType::F32)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
fn full_graph_training_moves_every_attention_site_and_ships_the_eval_contract() {
    let root = scratch("e2e");
    let args = args_for(&root);
    let summary = run(&args).unwrap();

    assert_eq!(summary.losses.len(), 4);
    assert!(summary.all_grads_finite);
    assert!(
        summary.losses.iter().all(|l| l.is_finite()),
        "{:?}",
        summary.losses
    );
    assert!(
        summary.losses.first().unwrap() > summary.losses.last().unwrap(),
        "four AdamW steps at lr 1e-2 must reduce the training CE, got {:?}",
        summary.losses
    );
    assert_eq!(summary.config_layers, 4);
    assert_eq!((summary.moe_layers, summary.dense_mlp_layers), (2, 2));
    assert_eq!(summary.num_examples, 2);
    assert_eq!(summary.num_eval_examples, 2);
    assert_eq!(
        summary.modules.len(),
        4 * 5,
        "four layers times q,k,v,o,g: {:?}",
        summary.modules
    );
    assert_eq!(summary.trainable_vars, 4 * 5 * 2);
    assert!(summary.adapter_path.is_file());
    assert!(summary.config_path.is_file());
    for (label, v) in [
        ("eval_ce_adapter", &summary.eval_ce_adapter),
        ("eval_ce_base", &summary.eval_ce_base),
    ] {
        assert_eq!(v.len(), 2, "{label}");
        assert!(v.iter().all(|x| x.is_finite()), "{label}: {v:?}");
    }
    let (a, b) = (
        summary.eval_loss_adapter.unwrap(),
        summary.eval_loss_base.unwrap(),
    );
    assert!(
        a != b,
        "a trained adapter must move the held-out CE off the base value ({a} vs {b})"
    );

    let weights = WeightLoader::open_file(&summary.adapter_path, &Device::Cpu).unwrap();
    let names = weights.names();
    let b_names: Vec<&String> = names.iter().filter(|n| n.ends_with("lora_B.weight")).collect();
    assert_eq!(b_names.len(), 4 * 5, "{b_names:?}");
    for name in &b_names {
        let t = weights.get(name, DType::F32).unwrap();
        assert!(
            amax(&t) > 0.0,
            "{name} is still exactly zero, so this module was unreachable from the loss; the \
             full-graph replica must keep q and k on the graph through \
             softmax_last_dim_that_keeps_q_and_k_on_the_graph"
        );
    }
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&summary.config_path).unwrap()).unwrap();
    let targets: Vec<String> = cfg["target_modules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    for leaf in ["q_proj", "k_proj", "v_proj", "o_proj", "g_proj"] {
        assert!(targets.contains(&leaf.to_string()), "{targets:?}");
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_trainer_replica_forward_matches_the_serving_laguna_forward_on_the_fixture() {
    let root = scratch("parity");
    let base = root.join("base");
    let cfg = write_tiny_laguna_checkpoint(&base);
    let device = Device::Cpu;

    let weights = WeightLoader::open_dir(&base, &device).unwrap();
    let serving = Laguna::from_loader(cfg, &weights, &device).unwrap();
    let train = LagunaTrainModel::load_with_dtype(&base, &device, DType::BF16).unwrap();

    let ids: Vec<u32> = (0..24u32).map(|i| (i * 17 + 3) % 96).collect();
    let seq = ids.len();
    let tokens = Tensor::from_vec(ids, (1usize, seq), &device).unwrap();
    let positions =
        Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), seq, &device).unwrap();

    let serving_logits = serving.forward(&tokens, &positions).unwrap();
    let train_logits = train.forward_logits(&tokens, &positions).unwrap();
    assert_eq!(serving_logits.dims(), train_logits.dims());
    let diff = amax(&(serving_logits.clone() - train_logits).unwrap());
    let scale = amax(&serving_logits);
    assert!(
        diff <= 0.05 * (1.0 + scale),
        "trainer forward drifted from the serving oracle: max abs diff {diff} against logit \
         scale {scale}; the replica must reproduce yarn partial rope, the attention factor on \
         the rotated slice, per-head softplus gating, the sliding window mask and host MoE \
         routing exactly"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_moe_expert_path_backward_matches_a_frozen_routing_finite_difference_probe() {
    let root = scratch("moe-grad");
    let base = root.join("base");
    let cfg = write_tiny_laguna_checkpoint(&base);
    let device = Device::Cpu;
    let model = LagunaTrainModel::load(&base, &device).unwrap();
    let moe = model
        .moe_of_the_first_sparse_layer_so_tests_can_gate_its_backward()
        .expect("the tiny fixture declares sparse mlp layers");

    let h = cfg.hidden_size;
    let n_tokens = 3usize;
    let x0: Vec<f32> = (0..n_tokens * h)
        .map(|i| ((i * 37 + 11) % 97) as f32 / 97.0 - 0.5)
        .collect();
    let u: Vec<f32> = (0..n_tokens * h)
        .map(|i| ((i * 53 + 29) % 101) as f32 / 101.0 - 0.5)
        .collect();
    let x_var = Var::from_vec(x0.clone(), (n_tokens, h), &device).unwrap();
    let u_t = Tensor::from_vec(u.clone(), (n_tokens, h), &device).unwrap();

    let loss = moe
        .forward(x_var.as_tensor())
        .unwrap()
        .mul(&u_t)
        .unwrap()
        .sum_all()
        .unwrap();
    let grads = loss.backward().unwrap();
    let gx = grads.get(&x_var).expect(
        "the MoE output left the autograd graph w.r.t. its input: a host round-trip or a lost \
         index_select/index_add backward detaches the expert path silently, and every \
         attention-site gate still passes through the residual add in forward_logits",
    );
    let gx_host: Vec<f32> = gx
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let x0_t = Tensor::from_vec(x0.clone(), (n_tokens, h), &device).unwrap();
    let (ids, weights) = moe.route_host(&x0_t, n_tokens).unwrap();
    let frozen_routing_loss = |xs: &[f32]| -> f32 {
        let mut total = 0f32;
        for t in 0..n_tokens {
            let x_row =
                Tensor::from_vec(xs[t * h..(t + 1) * h].to_vec(), (1usize, h), &device).unwrap();
            let mut y = vec![0f32; h];
            for j in 0..moe.top_k {
                let e = ids[t * moe.top_k + j] as usize;
                let w = weights[t * moe.top_k + j];
                let ye: Vec<f32> = moe.experts[e]
                    .forward(&x_row)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                for (acc, v) in y.iter_mut().zip(ye) {
                    *acc += w * v;
                }
            }
            for acc in y.iter_mut() {
                *acc *= moe.routed_scaling;
            }
            let shared: Vec<f32> = moe
                .shared_expert
                .forward(&x_row)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            for (i, (acc, sh)) in y.iter().zip(shared).enumerate() {
                total += (acc + sh) * u[t * h + i];
            }
        }
        total
    };

    let eps = 1e-2f32;
    let mut max_abs_ad = 0f32;
    for probe in 0..8usize {
        let i = (probe * 13 + 5) % (n_tokens * h);
        let mut xp = x0.clone();
        xp[i] += eps;
        let mut xm = x0.clone();
        xm[i] -= eps;
        let fd = (frozen_routing_loss(&xp) - frozen_routing_loss(&xm)) / (2.0 * eps);
        let ad = gx_host[i];
        max_abs_ad = max_abs_ad.max(ad.abs());
        assert!(
            (ad - fd).abs() <= 5e-3 + 0.05 * fd.abs(),
            "autograd d(loss)/dx[{i}] = {ad} disagrees with the frozen-routing central \
             difference {fd}: the expert path (index_select -> frozen Mlp -> index_add, \
             routing weights detached by design) is the only differentiable FFN route, so a \
             wrong or missing backward here trains attention sites on a silently wrong loss \
             surface"
        );
    }
    assert!(
        max_abs_ad > 1e-3,
        "every probed MoE input gradient is ~zero ({max_abs_ad}); the probe set no longer \
         exercises the expert path and this gate has gone vacuous"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn unsupported_targets_and_gating_free_bases_are_refused_with_the_reason() {
    let root = scratch("refuse-targets");
    let mut args = args_for(&root);
    args.targets = vec!["q".into(), "gate".into()];
    let err = format!("{:#}", run(&args).unwrap_err());
    assert!(
        err.contains("gate") && err.contains("attention projections"),
        "{err}"
    );

    let cfg_path = args.base.join("config.json");
    let mut cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&cfg_path).unwrap()).unwrap();
    cfg["gating"] = serde_json::Value::Bool(false);
    std::fs::write(&cfg_path, serde_json::to_string(&cfg).unwrap()).unwrap();
    args.targets = LagunaTrainArgs::default_targets();
    let err = format!("{:#}", run(&args).unwrap_err());
    assert!(
        err.contains("gating disabled") && err.contains("g_proj"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_packed_nvfp4_base_is_refused_before_any_training() {
    let root = scratch("refuse-packed");
    let mut args = args_for(&root);
    let packed_base = root.join("packed-base");
    std::fs::create_dir_all(&packed_base).unwrap();
    std::fs::copy(
        args.base.join("config.json"),
        packed_base.join("config.json"),
    )
    .unwrap();
    let mut t = std::collections::HashMap::new();
    t.insert(
        "model.layers.1.mlp.experts.0.gate_proj.weight_packed".to_string(),
        Tensor::zeros((4usize, 4usize), DType::F32, &Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&t, packed_base.join("model.safetensors")).unwrap();
    args.base = packed_base;
    let err = format!("{:#}", run(&args).unwrap_err());
    assert!(
        err.contains("weight_packed") && err.contains("nvfp4"),
        "the refusal must name the packed tensor form so the operator knows this is the \
         shipped NVFP4 checkpoint and not a corrupt download: {err}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_fixture_config_is_a_laguna_config_the_serving_loader_accepts() {
    let cfg = LagunaConfig::from_hf_json_str(&tiny_laguna_config_json()).unwrap();
    assert_eq!(cfg.num_hidden_layers, 4);
    assert_eq!(cfg.num_heads_for_layer(1), 8);
    assert!(!cfg.is_moe_layer(0) && cfg.is_moe_layer(1));
}
