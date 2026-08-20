#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::moe::{MoeBlock, MoeConfig};
mod common;
use common::cuda as pick_device;

fn make_linear(device: &Device, out_f: usize, in_f: usize) -> Linear {
    let w = Tensor::randn(0f32, 0.02, (out_f, in_f), device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    Linear::new(w, None).unwrap()
}

fn make_mlp(device: &Device, hidden: usize, intermediate: usize) -> Mlp {
    Mlp::new(
        make_linear(device, intermediate, hidden),
        make_linear(device, intermediate, hidden),
        make_linear(device, hidden, intermediate),
    )
    .unwrap()
}

fn build_random_moe(device: &Device, cfg: MoeConfig) -> MoeBlock {
    let gate = make_linear(device, cfg.num_experts, cfg.hidden_size);
    let experts = (0..cfg.num_experts)
        .map(|_| make_mlp(device, cfg.hidden_size, cfg.moe_intermediate_size))
        .collect();
    let shared_expert = make_mlp(device, cfg.hidden_size, cfg.shared_expert_intermediate_size);
    let shared_expert_gate = make_linear(device, 1, cfg.hidden_size);
    MoeBlock::new(cfg, gate, experts, shared_expert, shared_expert_gate).unwrap()
}

#[test]
fn moe_forward_shape_and_finite() {
    let Some(device) = pick_device() else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let cfg = MoeConfig {
        hidden_size: 64,
        num_experts: 8,
        num_experts_per_tok: 2,
        moe_intermediate_size: 32,
        shared_expert_intermediate_size: 32,
    };
    let moe = build_random_moe(&device, cfg);
    let x = Tensor::randn(0f32, 1.0, (1usize, 4usize, cfg.hidden_size), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let y = moe.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 4, cfg.hidden_size]);
    assert_eq!(y.dtype(), DType::BF16);

    let v: Vec<f32> = y
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "non-finite output");
    assert!(v.iter().any(|x| x.abs() > 1e-6), "moe produced all zeros");

    let x_v: Vec<f32> = x
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let differ = v.iter().zip(x_v.iter()).any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(differ, "moe output equals input -- forward is degenerate");
}

#[test]
fn moe_single_active_matches_chosen_expert_plus_shared() {
    let Some(device) = pick_device() else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let hidden = 16usize;
    let num_experts = 4usize;
    let inter = 24usize;
    let cfg = MoeConfig {
        hidden_size: hidden,
        num_experts,
        num_experts_per_tok: 1,
        moe_intermediate_size: inter,
        shared_expert_intermediate_size: inter,
    };

    let n_tokens = 4usize;
    let mut routed_gate_w = vec![0f32; num_experts * hidden];
    for e in 0..num_experts {
        routed_gate_w[e * hidden + (e % hidden)] = 1.0;
    }
    let gate_tensor = Tensor::from_vec(routed_gate_w, (num_experts, hidden), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let gate = Linear::new(gate_tensor, None).unwrap();

    let experts: Vec<Mlp> = (0..num_experts)
        .map(|_| make_mlp(&device, hidden, inter))
        .collect();
    let shared_expert = make_mlp(&device, hidden, inter);
    let shared_expert_gate = make_linear(&device, 1, hidden);

    let mut tokens = vec![0f32; n_tokens * hidden];
    for n in 0..n_tokens {
        let e = n % num_experts;
        tokens[n * hidden + (e % hidden)] = 10.0;
    }
    let x = Tensor::from_vec(tokens, (1usize, n_tokens, hidden), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let moe = MoeBlock::new(cfg, gate, experts, shared_expert, shared_expert_gate).unwrap();

    let y = moe.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, n_tokens, hidden]);

    let x_flat = x.reshape((n_tokens, hidden)).unwrap();
    let mut expected_rows: Vec<Tensor> = Vec::with_capacity(n_tokens);
    for n in 0..n_tokens {
        let e = n % num_experts;
        let row = x_flat.narrow(0, n, 1).unwrap();
        let expert_out = moe.expert(e).forward(&row).unwrap();
        expected_rows.push(expert_out);
    }
    let expert_part = Tensor::cat(&expected_rows, 0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let shared_out = moe
        .shared_expert()
        .forward(&x_flat)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let shared_gate_logits = moe
        .shared_expert_gate()
        .forward(&x_flat)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let shared_gate = candle_nn::ops::sigmoid(&shared_gate_logits).unwrap();
    let expected = expert_part
        .add(&shared_gate.broadcast_mul(&shared_out).unwrap())
        .unwrap();

    let got: Vec<f32> = y
        .to_dtype(DType::F32)
        .unwrap()
        .reshape((n_tokens, hidden))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let exp_v: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();
    let mut max_abs = 0f32;
    for (g, e) in got.iter().zip(exp_v.iter()) {
        max_abs = max_abs.max((g - e).abs());
    }
    assert!(
        max_abs < 5e-2,
        "single-active dispatch drift max_abs={max_abs}"
    );
}

#[test]
fn moe_deterministic() {
    let Some(device) = pick_device() else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let cfg = MoeConfig {
        hidden_size: 32,
        num_experts: 6,
        num_experts_per_tok: 2,
        moe_intermediate_size: 16,
        shared_expert_intermediate_size: 16,
    };
    let moe = build_random_moe(&device, cfg);
    let x = Tensor::randn(0f32, 1.0, (2usize, 3usize, cfg.hidden_size), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let y1 = moe.forward(&x).unwrap();
    let y2 = moe.forward(&x).unwrap();
    let v1: Vec<f32> = y1
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let v2: Vec<f32> = y2
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "moe non-deterministic");
    }
}

fn moe_skip(msg: &str) {
    if std::env::var("NV_MOE_CKPT_ALLOW_SKIP").as_deref() == Ok("1") {
        eprintln!(
            "SKIP (NV_MOE_CKPT_ALLOW_SKIP=1): moe_real_checkpoint_keys_present_if_available: \
             {msg}. This is a SKIP, not a pass -- no checkpoint key was checked."
        );
        return;
    }
    panic!(
        "moe_real_checkpoint_keys_present_if_available: {msg}. Set NV_MOE_CKPT_ALLOW_SKIP=1 to \
         skip on purpose."
    );
}

#[test]
fn moe_real_checkpoint_keys_present_if_available() {
    use std::path::Path;
    let repos = [
        "models--RedHatAI--Qwen3.6-35B-A3B-NVFP4",
        "models--RedHatAI--Qwen3.5-MoE-NVFP4",
    ];
    let mut roots: Vec<std::path::PathBuf> = Vec::new();
    if let Some(v) = std::env::var_os("HF_HUB_CACHE") {
        roots.push(std::path::PathBuf::from(v));
    }
    if let Some(h) = std::env::var_os("HOME") {
        roots.push(std::path::PathBuf::from(h).join(".cache/huggingface/hub"));
    }
    let mut snapshot_dir: Option<std::path::PathBuf> = None;
    'search: for root in &roots {
        for repo in &repos {
            let snapshots = Path::new(root).join(repo).join("snapshots");
            let mut cands: Vec<std::path::PathBuf> = std::fs::read_dir(&snapshots)
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    let has_lm_weights_so_this_is_not_a_vision_tower_only_snapshot = p
                        .join("model.safetensors.index.json")
                        .is_file()
                        || p.join("model.safetensors").is_file();
                    p.join("config.json").is_file()
                        && has_lm_weights_so_this_is_not_a_vision_tower_only_snapshot
                })
                .collect();
            cands.sort();
            if let Some(p) = cands.pop() {
                snapshot_dir = Some(p);
                break 'search;
            }
        }
    }
    let Some(dir) = snapshot_dir else {
        moe_skip("no Qwen3.5/3.6 MoE NVFP4 checkpoint found in the HF cache");
        return;
    };

    let device = match pick_device() {
        Some(d) => d,
        None => {
            moe_skip("no CUDA device");
            return;
        }
    };
    let loader = match nv_weights::WeightLoader::open_dir(&dir, &device) {
        Ok(l) => l,
        Err(e) => {
            moe_skip(&format!("cannot open loader: {e}"));
            return;
        }
    };

    let prefixes = [
        "model.language_model.layers.0.mlp",
        "model.layers.0.mlp",
        "language_model.model.layers.0.mlp",
    ];
    let mut prefix_hit: Option<&str> = None;
    for p in &prefixes {
        if loader.has(&format!("{p}.gate.weight"))
            || loader.has(&format!("{p}.experts.0.gate_proj.weight"))
            || loader.has(&format!("{p}.experts.0.gate_proj.weight_packed"))
        {
            prefix_hit = Some(*p);
            break;
        }
    }
    let Some(prefix) = prefix_hit else {
        panic!(
            "moe_real_checkpoint_keys_present_if_available: the checkpoint at {} matched NONE of \
             the MoE prefixes {prefixes:?}. This is the one guard that is NOT an availability \
             question -- the checkpoint IS open and loaded at this point, so a prefix miss means \
             the naming convention moved and every key assertion below it was skipped. It used to \
             print one line and report a pass.",
            dir.display()
        );
    };

    let must_have_router = format!("{prefix}.gate.weight");
    assert!(
        loader.has(&must_have_router),
        "router key missing: {must_have_router}"
    );

    let shared_router = format!("{prefix}.shared_expert_gate.weight");
    assert!(
        loader.has(&shared_router),
        "shared_expert_gate key missing: {shared_router}"
    );

    let mut expert0_seen = false;
    for suffix in &["weight", "weight_packed"] {
        if loader.has(&format!("{prefix}.experts.0.gate_proj.{suffix}")) {
            expert0_seen = true;
            break;
        }
    }
    assert!(expert0_seen, "expert 0 gate_proj key missing");

    let mut shared_seen = false;
    for suffix in &["weight", "weight_packed"] {
        if loader.has(&format!("{prefix}.shared_expert.gate_proj.{suffix}")) {
            shared_seen = true;
            break;
        }
    }
    assert!(shared_seen, "shared_expert.gate_proj key missing");
}
