#![cfg(feature = "wgpu")]

mod common;
use common::argmax_partial_cmp as argmax;
use common::have_gpu;
use common::HIDDEN_64 as HIDDEN;
use common::LcgTop24TwoSided as Lcg;
use common::N_EXPERTS;
use common::norm_tensor;
use common::nozi_prof_dump;
use common::rand_tensor_f32_shape as rand_tensor;
use common::real_snapshot;
use common::rel_err;
use common::tiny_config_json;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4_moe::{Gemma4Moe, Gemma4MoeConfig};
use nv_models::gemma4_moe_wgpu::{
    dequantize_w4_expert, host_layer_from_loader, Gemma4MoeWgpu, HostW4Stack,
};
use nv_weights::WeightLoader;
use std::collections::HashMap;
use common::TempDir;
use common::tensors_for_cfg;

fn tiny_tensors(seed: u64) -> HashMap<String, Tensor> {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    tensors_for_cfg(&cfg, seed)
}

fn real_heads_config_json() -> String {
    r#"{
  "model_type": "gemma4",
  "tie_word_embeddings": true,
  "text_config": {
    "attention_k_eq_v": true,
    "enable_moe_block": true,
    "final_logit_softcapping": 30.0,
    "global_head_dim": 512,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 64,
    "intermediate_size": 96,
    "layer_types": ["sliding_attention", "full_attention"],
    "max_position_embeddings": 64,
    "moe_intermediate_size": 64,
    "num_attention_heads": 2,
    "num_experts": 8,
    "num_global_key_value_heads": 1,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
      "sliding_attention": {"rope_theta": 10000.0}
    },
    "sliding_window": 4,
    "tie_word_embeddings": true,
    "top_k_experts": 2,
    "vocab_size": 160
  }
}"#
    .to_string()
}

fn temp_dir(tag: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("g4m_wgpu_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

fn dequant_stack_rows(stack: &HostW4Stack, e: usize) -> Vec<f32> {
    dequantize_w4_expert(stack, e)
}

fn write_dequantized_oracle_file(
    cfg: &Gemma4MoeConfig,
    src: &std::path::Path,
    tensors: &HashMap<String, Tensor>,
    dst: &std::path::Path,
) {
    let loader = WeightLoader::open_file(src, &Device::Cpu).unwrap();
    let mut out = tensors.clone();
    for i in 0..cfg.base.num_hidden_layers {
        let layer = host_layer_from_loader(cfg, &loader, i).unwrap();
        let mi = cfg.moe_intermediate_size;
        let hidden = cfg.base.hidden_size;
        let n_e = cfg.num_experts;
        let mut gu = vec![0f32; n_e * 2 * mi * hidden];
        let mut dn = vec![0f32; n_e * hidden * mi];
        for e in 0..n_e {
            let gate = dequant_stack_rows(&layer.experts_gate, e);
            let up = dequant_stack_rows(&layer.experts_up, e);
            let down = dequant_stack_rows(&layer.experts_down, e);
            let base = e * 2 * mi * hidden;
            gu[base..base + mi * hidden].copy_from_slice(&gate);
            gu[base + mi * hidden..base + 2 * mi * hidden].copy_from_slice(&up);
            dn[e * hidden * mi..(e + 1) * hidden * mi].copy_from_slice(&down);
        }
        let p = format!("model.language_model.layers.{i}");
        out.insert(
            format!("{p}.experts.gate_up_proj"),
            Tensor::from_vec(gu, (n_e, 2 * mi, hidden), &Device::Cpu).unwrap(),
        );
        out.insert(
            format!("{p}.experts.down_proj"),
            Tensor::from_vec(dn, (n_e, hidden, mi), &Device::Cpu).unwrap(),
        );
    }
    candle_core::safetensors::save(&out, dst).unwrap();
}

#[test]
fn real_config_is_supported_by_the_wgpu_module() {
    let real = include_str!("gemma4_moe_wgpu_real_config.json");
    let cfg = Gemma4MoeConfig::from_hf_json_str(real).unwrap();
    assert!(cfg.base.hidden_size.is_multiple_of(32));
    assert!(cfg.moe_intermediate_size.is_multiple_of(64));
    assert!(cfg.num_experts <= 256);
    assert!(cfg.top_k_experts <= 16);
    assert!(cfg.base.head_dim <= 512 && cfg.base.head_dim.is_multiple_of(2));
    assert!(cfg.base.global_head_dim <= 512 && cfg.base.global_head_dim.is_multiple_of(2));
    assert!(cfg.base.tie_word_embeddings);
    assert!(cfg
        .base
        .num_attention_heads
        .is_multiple_of(cfg.base.num_key_value_heads));
    assert!(cfg
        .base
        .num_attention_heads
        .is_multiple_of(cfg.base.num_global_key_value_heads.unwrap()));
}

fn compare_gpu_vs_oracle(
    cfg: &Gemma4MoeConfig,
    seed: u64,
    tag: &str,
    tokens: &[u32],
    max_rel: f32,
) {
    let dir = temp_dir(tag);
    let st_a = dir.0.join("model.safetensors");
    let st_b = dir.0.join("oracle.safetensors");
    let tensors = tensors_for_cfg(cfg, seed);
    candle_core::safetensors::save(&tensors, &st_a).unwrap();
    write_dequantized_oracle_file(cfg, &st_a, &tensors, &st_b);

    let loader_a = WeightLoader::open_file(&st_a, &Device::Cpu).unwrap();
    let mut gpu = Gemma4MoeWgpu::from_loader(cfg.clone(), &loader_a, 32).expect("build wgpu model");
    eprintln!("[{tag}] recorded passes per token: {}", gpu.pass_count());

    let loader_b = WeightLoader::open_file(&st_b, &Device::Cpu).unwrap();
    let oracle =
        Gemma4Moe::from_loader_dtype(cfg.clone(), &loader_b, &Device::Cpu, DType::BF16).unwrap();
    let mut cache = oracle.new_kv_cache(32).unwrap();

    let mut worst_rel = 0f32;
    let mut top1_hits = 0usize;
    let mut all_logits: Vec<Vec<f32>> = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let (arg, logits) = gpu.decode_step_logits(*t).expect("decode step");
        all_logits.push(logits.clone());
        let tok_t = Tensor::from_vec(vec![*t], (1usize, 1usize), &Device::Cpu).unwrap();
        let pos_t = Tensor::from_vec(vec![i as i32], 1usize, &Device::Cpu).unwrap();
        let want: Vec<f32> = oracle
            .forward_with_cache(&tok_t, &pos_t, &mut cache)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let (abs, rel) = rel_err(&logits, &want);
        let ref_arg = argmax(&want);
        if arg == ref_arg {
            top1_hits += 1;
        }
        worst_rel = worst_rel.max(rel);
        eprintln!(
            "[{tag}] step {i}: tok={t} gpu_argmax={arg} ref_argmax={ref_arg} max_abs={abs:.6} rel={rel:.6}"
        );
        assert!(
            rel < max_rel,
            "[{tag}] step {i}: logits diverged from the candle oracle (rel {rel})"
        );
    }
    eprintln!(
        "[{tag}] worst relative logit error over {} steps: {worst_rel:.6}",
        tokens.len()
    );
    assert_eq!(
        top1_hits,
        tokens.len(),
        "[{tag}] argmax disagreed with the oracle on {} of {} steps",
        tokens.len() - top1_hits,
        tokens.len()
    );

    let (spread, _) = rel_err(&all_logits[0], &all_logits[2]);
    assert!(
        spread > 1e-3,
        "[{tag}] logits are insensitive to token / KV state (spread {spread})"
    );
}

#[test]
fn tiny_wgpu_decode_matches_dequantized_candle_oracle() {
    if !have_gpu() {
        return;
    }
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let tokens: [u32; 12] = [3, 11, 5, 40, 2, 19, 77, 4, 120, 8, 55, 33];
    compare_gpu_vs_oracle(&cfg, 0x9e37_79b9_7f4a_7c15, "tiny", &tokens, 0.05);
}

#[test]
fn real_head_geometry_matches_oracle() {
    if !have_gpu() {
        return;
    }
    let cfg = Gemma4MoeConfig::from_hf_json_str(&real_heads_config_json()).unwrap();
    assert_eq!(cfg.base.global_head_dim, 512);
    assert_eq!(cfg.base.head_dim, 256);
    let tokens: [u32; 8] = [3, 11, 5, 40, 2, 19, 77, 4];
    compare_gpu_vs_oracle(&cfg, 0x51ee_d100_0004, "hd512", &tokens, 0.05);
}

#[test]
fn tiny_wgpu_is_deterministic_across_reset_and_bounds_kv() {
    if !have_gpu() {
        return;
    }
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let dir = temp_dir("det");
    let st = dir.0.join("model.safetensors");
    candle_core::safetensors::save(&tiny_tensors(0x51ee_d100_0002), &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
    let mut gpu = Gemma4MoeWgpu::from_loader(cfg.clone(), &loader, 4).expect("build");
    assert!(gpu.pass_count() > 0);
    assert!(gpu.vram_report().total_bytes > 0);

    let tokens = [7u32, 3, 9, 12];
    let mut run1: Vec<(u32, Vec<f32>)> = Vec::new();
    for t in tokens {
        run1.push(gpu.decode_step_logits(t).expect("step"));
    }
    assert_eq!(gpu.current_pos(), 4);
    let err = gpu.decode_step(1).unwrap_err();
    assert!(
        err.to_string().contains("kv cache full"),
        "expected kv-full error, got: {err}"
    );

    gpu.reset().expect("reset");
    assert_eq!(gpu.current_pos(), 0);
    for (i, t) in tokens.iter().enumerate() {
        let (arg, logits) = gpu.decode_step_logits(*t).expect("step");
        assert_eq!(arg, run1[i].0, "argmax changed across reset at step {i}");
        assert!(
            logits
                .iter()
                .zip(run1[i].1.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "logits are not bit-identical across reset at step {i}"
        );
    }
    eprintln!(
        "[wgpu] determinism across reset verified over {} steps",
        tokens.len()
    );

    let (a0, l0) = (run1[0].0, &run1[0].1);
    let (a1, l1) = (run1[1].0, &run1[1].1);
    let differs = a0 != a1 || l0.iter().zip(l1.iter()).any(|(x, y)| (x - y).abs() > 1e-6);
    assert!(differs, "KV state is not carried between steps");
}

#[test]
fn tiny_from_loader_rejects_bad_shapes() {
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let dir = temp_dir("shapes");
    let st = dir.0.join("model.safetensors");
    let mut tensors = tiny_tensors(0x51ee_d100_0003);
    tensors.insert(
        "model.language_model.layers.0.router.proj.weight".into(),
        Tensor::zeros((N_EXPERTS, HIDDEN + 2), DType::F32, &Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&tensors, &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
    let err = match host_layer_from_loader(&cfg, &loader, 0) {
        Ok(_) => panic!("expected a shape error"),
        Err(e) => e,
    };
    assert!(
        format!("{err:#}").contains("router.proj.weight"),
        "unexpected error: {err:#}"
    );
}

fn hand_logits_no_layers(cfg: &Gemma4MoeConfig, loader: &WeightLoader, tok: u32) -> Vec<f32> {
    let hidden = cfg.base.hidden_size;
    let vocab = cfg.base.vocab_size;
    let embed = loader
        .get("model.language_model.embed_tokens.weight", DType::F32)
        .unwrap();
    let wf: Vec<f32> = loader
        .get("model.language_model.norm.weight", DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let e: Vec<f32> = embed
        .narrow(0, tok as usize, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let scale = (hidden as f32).sqrt();
    let x: Vec<f32> = e.iter().map(|v| v * scale).collect();
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
    let inv = 1.0 / (mean_sq + cfg.base.rms_norm_eps as f32).sqrt();
    let h: Vec<f32> = x.iter().zip(wf.iter()).map(|(v, w)| v * inv * w).collect();
    let flat: Vec<f32> = embed.flatten_all().unwrap().to_vec1().unwrap();
    let cap = cfg.base.final_logit_softcapping;
    (0..vocab)
        .map(|t| {
            let row = &flat[t * hidden..(t + 1) * hidden];
            let raw: f32 = row.iter().zip(h.iter()).map(|(a, b)| a * b).sum();
            if cap > 0.0 {
                (raw / cap).tanh() * cap
            } else {
                raw
            }
        })
        .collect()
}

fn hand_attn_only_logits(cfg: &Gemma4MoeConfig, loader: &WeightLoader, tok: u32) -> Vec<f32> {
    let hidden = cfg.base.hidden_size;
    let vocab = cfg.base.vocab_size;
    let eps = cfg.base.rms_norm_eps as f32;
    let getv = |name: &str| -> Vec<f32> {
        loader
            .get(name, DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let rms_norm = |x: &[f32], w: &[f32]| -> Vec<f32> {
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        x.iter().zip(w.iter()).map(|(v, g)| v * inv * g).collect()
    };
    let matvec = |w: &[f32], x: &[f32], n: usize, k: usize| -> Vec<f32> {
        (0..n)
            .map(|r| {
                w[r * k..(r + 1) * k]
                    .iter()
                    .zip(x.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    };
    let p = "model.language_model.layers.0";
    let embed = getv("model.language_model.embed_tokens.weight");
    let scale = (hidden as f32).sqrt();
    let x: Vec<f32> = embed[tok as usize * hidden..(tok as usize + 1) * hidden]
        .iter()
        .map(|v| v * scale)
        .collect();
    let normed = rms_norm(&x, &getv(&format!("{p}.input_layernorm.weight")));
    let kind = cfg.base.layer_kind(0);
    let hd = cfg.base.head_dim_for(kind);
    let n_kv = cfg.base.num_kv_heads_for(kind);
    let n_q = cfg.base.num_attention_heads;
    let vname = format!("{p}.self_attn.v_proj.weight");
    let v_w = if loader.has(&vname) {
        getv(&vname)
    } else {
        getv(&format!("{p}.self_attn.k_proj.weight"))
    };
    let v = matvec(&v_w, &normed, n_kv * hd, hidden);
    let ones = vec![1.0f32; hd];
    let mut vn = vec![0f32; n_kv * hd];
    for h in 0..n_kv {
        let out = rms_norm(&v[h * hd..(h + 1) * hd], &ones);
        vn[h * hd..(h + 1) * hd].copy_from_slice(&out);
    }
    let group = n_q / n_kv;
    let mut o_in = vec![0f32; n_q * hd];
    for h in 0..n_q {
        let kv = h / group;
        o_in[h * hd..(h + 1) * hd].copy_from_slice(&vn[kv * hd..(kv + 1) * hd]);
    }
    let attn = matvec(
        &getv(&format!("{p}.self_attn.o_proj.weight")),
        &o_in,
        hidden,
        n_q * hd,
    );
    let attn_post = rms_norm(
        &attn,
        &getv(&format!("{p}.post_attention_layernorm.weight")),
    );
    let ls = getv(&format!("{p}.layer_scalar"))[0];
    let h_out: Vec<f32> = x
        .iter()
        .zip(attn_post.iter())
        .map(|(a, b)| (a + b) * ls)
        .collect();
    let final_x = rms_norm(&h_out, &getv("model.language_model.norm.weight"));
    let cap = cfg.base.final_logit_softcapping;
    (0..vocab)
        .map(|t| {
            let raw: f32 = embed[t * hidden..(t + 1) * hidden]
                .iter()
                .zip(final_x.iter())
                .map(|(a, b)| a * b)
                .sum();
            if cap > 0.0 {
                (raw / cap).tanh() * cap
            } else {
                raw
            }
        })
        .collect()
}

#[test]
#[ignore = "loads 2 real layers on CPU and GPU; set NV_GEMMA4_MOE_WGPU_TEST=1"]
fn real_truncated_two_layers_gpu_matches_oracle() {
    if std::env::var("NV_GEMMA4_MOE_WGPU_TEST").as_deref() != Ok("1") {
        eprintln!("[skip] NV_GEMMA4_MOE_WGPU_TEST not set");
        return;
    }
    if !have_gpu() {
        panic!("needs a wgpu adapter");
    }
    let snap = real_snapshot();
    let mut cfg = Gemma4MoeConfig::from_hf_json_file(&snap.join("config.json")).unwrap();
    let n_layers: usize = std::env::var("NV_GEMMA4_MOE_TRUNC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    cfg.base.num_hidden_layers = n_layers;
    cfg.base.layer_types.truncate(n_layers);
    cfg.base.final_logit_softcapping = 0.0;

    let ablate = std::env::var("NV_GEMMA4_MOE_ABLATE").unwrap_or_default();
    let loader = WeightLoader::open_dir(&snap, &Device::Cpu).expect("open safetensors");

    let dir = temp_dir("real_trunc");
    let st_a = dir.0.join("model.safetensors");
    {
        let mut out: HashMap<String, Tensor> = HashMap::new();
        let names = [
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
        ];
        for n in names {
            out.insert(
                n.to_string(),
                loader.get(n, candle_core::DType::BF16).unwrap(),
            );
        }
        for i in 0..n_layers {
            let p = format!("model.language_model.layers.{i}");
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "post_feedforward_layernorm_1.weight",
                "pre_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm_2.weight",
                "layer_scalar",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "router.proj.weight",
                "router.scale",
                "router.per_expert_scale",
                "experts.gate_up_proj",
                "experts.down_proj",
            ] {
                let name = format!("{p}.{suffix}");
                out.insert(
                    name.clone(),
                    loader.get(&name, candle_core::DType::BF16).unwrap(),
                );
            }
            let vname = format!("{p}.self_attn.v_proj.weight");
            if loader.has(&vname) {
                out.insert(
                    vname.clone(),
                    loader.get(&vname, candle_core::DType::BF16).unwrap(),
                );
            }
            let zero = |out: &mut HashMap<String, Tensor>, name: String| {
                let t = out.get(&name).unwrap();
                let z = Tensor::zeros(t.dims(), t.dtype(), &Device::Cpu).unwrap();
                out.insert(name, z);
            };
            for part in ablate.split(',') {
                match part {
                    "layer" => zero(&mut out, format!("{p}.layer_scalar")),
                    "experts" => {
                        zero(&mut out, format!("{p}.experts.gate_up_proj"));
                        zero(&mut out, format!("{p}.experts.down_proj"));
                    }
                    "mlp" => {
                        zero(&mut out, format!("{p}.mlp.gate_proj.weight"));
                        zero(&mut out, format!("{p}.mlp.up_proj.weight"));
                        zero(&mut out, format!("{p}.mlp.down_proj.weight"));
                    }
                    "attn" => zero(&mut out, format!("{p}.self_attn.o_proj.weight")),
                    _ => {}
                }
            }
        }
        candle_core::safetensors::save(&out, &st_a).unwrap();
    }
    if !ablate.is_empty() {
        eprintln!("[trunc] ABLATION active: {ablate}");
    }

    let loader = WeightLoader::open_file(&st_a, &Device::Cpu).expect("open truncated file");
    let mut gpu = Gemma4MoeWgpu::from_loader(cfg.clone(), &loader, 16).expect("build gpu");

    let st_b = dir.0.join("oracle.safetensors");
    {
        let mut out: HashMap<String, Tensor> = HashMap::new();
        for name in loader.names() {
            out.insert(
                name.clone(),
                loader.get(&name, candle_core::DType::BF16).unwrap(),
            );
        }
        for i in 0..n_layers {
            let p = format!("model.language_model.layers.{i}");
            let layer = host_layer_from_loader(&cfg, &loader, i).unwrap();
            let (hidden, mi, n_e) = (
                cfg.base.hidden_size,
                cfg.moe_intermediate_size,
                cfg.num_experts,
            );
            let mut gu = vec![0f32; n_e * 2 * mi * hidden];
            let mut dn = vec![0f32; n_e * hidden * mi];
            for e in 0..n_e {
                let gate = dequant_stack_rows(&layer.experts_gate, e);
                let up = dequant_stack_rows(&layer.experts_up, e);
                let down = dequant_stack_rows(&layer.experts_down, e);
                let base = e * 2 * mi * hidden;
                gu[base..base + mi * hidden].copy_from_slice(&gate);
                gu[base + mi * hidden..base + 2 * mi * hidden].copy_from_slice(&up);
                dn[e * hidden * mi..(e + 1) * hidden * mi].copy_from_slice(&down);
            }
            out.insert(
                format!("{p}.experts.gate_up_proj"),
                Tensor::from_vec(gu, (n_e, 2 * mi, hidden), &Device::Cpu).unwrap(),
            );
            out.insert(
                format!("{p}.experts.down_proj"),
                Tensor::from_vec(dn, (n_e, hidden, mi), &Device::Cpu).unwrap(),
            );
        }
        candle_core::safetensors::save(&out, &st_b).unwrap();
    }

    let loader_b = WeightLoader::open_file(&st_b, &Device::Cpu).unwrap();
    let oracle =
        Gemma4Moe::from_loader_dtype(cfg.clone(), &loader_b, &Device::Cpu, DType::BF16).unwrap();
    let mut cache = oracle.new_kv_cache(16).unwrap();

    let tokens = [2u32, 818, 3468, 529, 6081, 563];
    let mut worst_rel = 0f32;
    for (i, t) in tokens.iter().enumerate() {
        let (arg, logits) = gpu.decode_step_logits(*t).expect("decode step");
        if i == 0 && ablate == "mlp,experts" && n_layers == 1 {
            let hand = hand_attn_only_logits(&cfg, &loader, *t);
            let tok_t0 = Tensor::from_vec(vec![*t], (1usize, 1usize), &Device::Cpu).unwrap();
            let pos_t0 = Tensor::from_vec(vec![0i32], 1usize, &Device::Cpu).unwrap();
            let mut c0 = oracle.new_kv_cache(16).unwrap();
            let ow: Vec<f32> = oracle
                .forward_with_cache(&tok_t0, &pos_t0, &mut c0)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let (_, rel_gpu) = rel_err(&logits, &hand);
            let (_, rel_or) = rel_err(&ow, &hand);
            eprintln!(
                "[trunc] attn-only hand check: gpu-vs-hand rel={rel_gpu:.6} \
                 oracle-vs-hand rel={rel_or:.6} hand_argmax={}",
                argmax(&hand)
            );
        }
        if i == 0 && ablate.contains("attn") && ablate.contains("layer") {
            let hand = hand_logits_no_layers(&cfg, &loader, *t);
            let tok_t0 = Tensor::from_vec(vec![*t], (1usize, 1usize), &Device::Cpu).unwrap();
            let pos_t0 = Tensor::from_vec(vec![0i32], 1usize, &Device::Cpu).unwrap();
            let mut c0 = oracle.new_kv_cache(16).unwrap();
            let ow: Vec<f32> = oracle
                .forward_with_cache(&tok_t0, &pos_t0, &mut c0)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let (_, rel_gpu) = rel_err(&logits, &hand);
            let (_, rel_or) = rel_err(&ow, &hand);
            eprintln!(
                "[trunc] hand-math check: gpu-vs-hand rel={rel_gpu:.6} oracle-vs-hand rel={rel_or:.6} \
                 hand_argmax={} gpu_argmax={} oracle_argmax={}",
                argmax(&hand),
                argmax(&logits),
                argmax(&ow)
            );
        }
        let tok_t = Tensor::from_vec(vec![*t], (1usize, 1usize), &Device::Cpu).unwrap();
        let pos_t = Tensor::from_vec(vec![i as i32], 1usize, &Device::Cpu).unwrap();
        let want: Vec<f32> = oracle
            .forward_with_cache(&tok_t, &pos_t, &mut cache)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let (abs, rel) = rel_err(&logits, &want);
        let ref_arg = argmax(&want);
        eprintln!(
            "[trunc] step {i}: tok={t} gpu_argmax={arg} ref_argmax={ref_arg} \
             max_abs={abs:.6} rel={rel:.6}"
        );
        worst_rel = worst_rel.max(rel);
        assert!(
            rel < 0.05,
            "[trunc] step {i}: GPU diverged from oracle on real weights (rel {rel})"
        );
        assert_eq!(arg, ref_arg, "[trunc] step {i}: argmax diverged");
    }
    eprintln!(
        "[trunc] worst rel over {} steps: {worst_rel:.6}",
        tokens.len()
    );
}

#[test]
#[ignore = "fetches nothing but loads ~17 GB wired; set NV_GEMMA4_MOE_WGPU_TEST=1"]
fn real_26b_a4b_wgpu_decode() {
    if std::env::var("NV_GEMMA4_MOE_WGPU_TEST").as_deref() != Ok("1") {
        eprintln!("[skip] NV_GEMMA4_MOE_WGPU_TEST not set");
        return;
    }
    if !have_gpu() {
        panic!("real-weights test needs a wgpu adapter");
    }
    let snap = match std::env::var("NV_GEMMA4_MOE_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            let root = std::path::PathBuf::from(std::env::var("HOME").unwrap())
                .join(".cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots");
            std::fs::read_dir(&root)
                .unwrap_or_else(|e| panic!("no snapshot dir at {}: {e}", root.display()))
                .filter_map(|d| d.ok())
                .map(|d| d.path())
                .find(|p| p.join("config.json").exists())
                .expect("no snapshot with config.json")
        }
    };
    let mut cfg = Gemma4MoeConfig::from_hf_json_file(&snap.join("config.json")).unwrap();
    if let Some(n) = std::env::var("NV_GEMMA4_MOE_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        assert!(n > 0 && n <= cfg.base.num_hidden_layers);
        cfg.base.num_hidden_layers = n;
        cfg.base.layer_types.truncate(n);
        eprintln!("[real] TRUNCATED to the first {n} layers (partial load)");
    }
    let max_seq: usize = std::env::var("NV_GEMMA4_MOE_MAX_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let loader = WeightLoader::open_dir(&snap, &Device::Cpu).expect("open safetensors");
    let t0 = std::time::Instant::now();
    let mut gpu = Gemma4MoeWgpu::from_loader(cfg.clone(), &loader, max_seq).expect("build");
    let report = gpu.load_report();
    let wired_gib = report.wired_bytes as f64 / (1u64 << 30) as f64;
    eprintln!(
        "[real] built in {:.1}s (quantize+host {:.1}s), {} passes/token, wired {wired_gib:.2} GiB",
        t0.elapsed().as_secs_f64(),
        report.quantize_s,
        gpu.pass_count()
    );
    assert!(
        wired_gib < 28.0,
        "wired device buffers {wired_gib:.2} GiB exceed the macOS working-set budget"
    );

    let tok = tokenizers::Tokenizer::from_file(snap.join("tokenizer.json")).expect("tokenizer");
    let prompt = std::env::var("NV_GEMMA4_MOE_PROMPT")
        .unwrap_or_else(|_| "The capital of France is".to_string());
    let enc = tok.encode(prompt.as_str(), false).expect("encode");
    let mut ids: Vec<u32> = enc.get_ids().to_vec();
    ids.insert(0, 2);
    let n_new: usize = std::env::var("NV_GEMMA4_MOE_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);

    let generate = |gpu: &mut Gemma4MoeWgpu| -> (Vec<u32>, f64) {
        let mut next = gpu.prefill(&ids).expect("prefill");
        let mut out = vec![next];
        let t1 = std::time::Instant::now();
        for _ in 1..n_new {
            next = gpu.decode_step(next).expect("decode");
            out.push(next);
        }
        let ms = t1.elapsed().as_secs_f64() * 1000.0 / (n_new.saturating_sub(1).max(1)) as f64;
        (out, ms)
    };

    let (out1, ms1) = generate(&mut gpu);
    let text = tok.decode(&out1, false).unwrap_or_default();
    eprintln!("[real] prompt={prompt:?}");
    eprintln!("[real] token_ids={out1:?}");
    eprintln!("[real] continuation={text:?}");
    eprintln!("[real] {ms1:.2} ms/tok decode");
    nozi_prof_dump();

    gpu.reset().expect("reset");
    let (out2, ms2) = generate(&mut gpu);
    eprintln!("[real] second run {ms2:.2} ms/tok");
    assert_eq!(out1, out2, "greedy decode is not reproducible across reset");

    if std::env::var("NV_GEMMA4_MOE_LAYERS").is_ok() {
        eprintln!("[real] truncated model: coherence is NOT expected, only that it runs");
        return;
    }
    assert!(
        out1.iter().any(|t| *t != out1[0]),
        "generation collapsed to a single repeated token"
    );
}
