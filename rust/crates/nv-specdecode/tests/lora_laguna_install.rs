#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use nv_models::laguna::{Laguna, LagunaConfig, LagunaFfn};
use nv_specdecode::lora_spec::{
    detach_laguna_serving_adapter, install_laguna_serving_adapter,
    LAGUNA_SERVING_LORA_PREFIX_IS_THE_GEMMA4_NAMESPACE_BECAUSE_NV_WEIGHTS_NORMALIZE_MODULE_NAME_REWRITES_MODEL_LAYERS,
};
use nv_weights::lora_adapter::LoraAdapter;
use nv_weights::{QuantScheme, QuantizationConfig, WeightLoader};

static ENV_AND_MODEL_BUILDS_RUN_ONE_AT_A_TIME_BECAUSE_FROM_LOADER_READS_NV_LAGUNA_ENV_KEYS:
    std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serialize() -> std::sync::MutexGuard<'static, ()> {
    ENV_AND_MODEL_BUILDS_RUN_ONE_AT_A_TIME_BECAUSE_FROM_LOADER_READS_NV_LAGUNA_ENV_KEYS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

const TINY_CONFIG: &str = r#"{
    "architectures": ["LagunaForCausalLM"],
    "model_type": "laguna",
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "num_experts": 2,
    "num_experts_per_tok": 1,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 16,
    "norm_topk_prob": true,
    "mlp_only_layers": [0],
    "decoder_sparse_step": 1,
    "tie_word_embeddings": false,
    "gating": "per-head",
    "sliding_window": 8,
    "moe_routed_scaling_factor": 2.5,
    "eos_token_id": [2],
    "rope_parameters": {
        "full_attention": {
            "rope_theta": 500000.0,
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 64,
            "beta_slow": 1.0,
            "beta_fast": 64.0,
            "attention_factor": 1.3465735902799727,
            "partial_rotary_factor": 0.5
        },
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0
        }
    },
    "layer_types": ["full_attention", "sliding_attention"],
    "mlp_layer_types": ["dense", "sparse"],
    "num_attention_heads_per_layer": [4, 4]
}"#;

fn scratch(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lora-laguna-{label}-{}-{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn det_vals(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.618 + seed) * 0.371).sin() * 0.15)
        .collect()
}

fn write_tiny_checkpoint(dir: &Path, gating_per_head: bool) {
    let cpu = Device::Cpu;
    let t = |out: usize, inp: usize, seed: f32| -> Tensor {
        Tensor::from_vec(det_vals(out * inp, seed), (out, inp), &cpu).unwrap()
    };
    let v = |n: usize| -> Tensor { Tensor::from_vec(vec![1.0f32; n], n, &cpu).unwrap() };
    let (h, hd, n_q, n_kv, inter, moe_i, vocab, experts) = (32, 8, 4, 2, 64, 16, 128, 2);
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert("model.embed_tokens.weight".into(), t(vocab, h, 7.0));
    tensors.insert("model.norm.weight".into(), v(h));
    tensors.insert("lm_head.weight".into(), t(vocab, h, 8.0));
    for i in 0..2usize {
        let p = format!("model.layers.{i}");
        let seed = i as f32 * 100.0;
        tensors.insert(format!("{p}.input_layernorm.weight"), v(h));
        tensors.insert(format!("{p}.post_attention_layernorm.weight"), v(h));
        tensors.insert(format!("{p}.self_attn.q_proj.weight"), t(n_q * hd, h, seed + 1.0));
        tensors.insert(format!("{p}.self_attn.k_proj.weight"), t(n_kv * hd, h, seed + 2.0));
        tensors.insert(format!("{p}.self_attn.v_proj.weight"), t(n_kv * hd, h, seed + 3.0));
        tensors.insert(format!("{p}.self_attn.o_proj.weight"), t(h, n_q * hd, seed + 4.0));
        if gating_per_head {
            tensors.insert(format!("{p}.self_attn.g_proj.weight"), t(n_q, h, seed + 5.0));
        }
        tensors.insert(format!("{p}.self_attn.q_norm.weight"), v(hd));
        tensors.insert(format!("{p}.self_attn.k_norm.weight"), v(hd));
        if i == 0 {
            tensors.insert(format!("{p}.mlp.gate_proj.weight"), t(inter, h, seed + 20.0));
            tensors.insert(format!("{p}.mlp.up_proj.weight"), t(inter, h, seed + 30.0));
            tensors.insert(format!("{p}.mlp.down_proj.weight"), t(h, inter, seed + 40.0));
        } else {
            tensors.insert(format!("{p}.mlp.gate.weight"), t(experts, h, seed + 50.0));
            for e in 0..experts {
                let ep = format!("{p}.mlp.experts.{e}");
                let es = seed + e as f32;
                tensors.insert(format!("{ep}.gate_proj.weight"), t(moe_i, h, es + 20.0));
                tensors.insert(format!("{ep}.up_proj.weight"), t(moe_i, h, es + 30.0));
                tensors.insert(format!("{ep}.down_proj.weight"), t(h, moe_i, es + 40.0));
            }
            let sp = format!("{p}.mlp.shared_expert");
            tensors.insert(format!("{sp}.gate_proj.weight"), t(moe_i, h, seed + 60.0));
            tensors.insert(format!("{sp}.up_proj.weight"), t(moe_i, h, seed + 70.0));
            tensors.insert(format!("{sp}.down_proj.weight"), t(h, moe_i, seed + 80.0));
        }
    }
    candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
    let cfg = if gating_per_head {
        TINY_CONFIG.to_string()
    } else {
        TINY_CONFIG.replace("\"per-head\"", "null")
    };
    std::fs::write(dir.join("config.json"), cfg).unwrap();
}

fn build_tiny_model(dir: &Path, device: &Device) -> Laguna {
    let raw = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let cfg = LagunaConfig::from_hf_json_str(&raw).unwrap();
    let weights = WeightLoader::open_dir(dir, device).unwrap();
    let qconfig = QuantizationConfig {
        scheme: QuantScheme::None,
        ignored_modules: Vec::new(),
        group_size: None,
    };
    Laguna::from_loader_quantized(cfg, &weights, &qconfig, device).unwrap()
}

fn write_adapter(dir: &Path, modules: &[(&str, usize, usize)], rank: usize) {
    std::fs::create_dir_all(dir).unwrap();
    let cpu = Device::Cpu;
    let cfg = serde_json::json!({
        "r": rank,
        "lora_alpha": (2 * rank) as f64,
        "target_modules": modules.iter().map(|(m, _, _)| *m).collect::<Vec<_>>(),
    });
    std::fs::write(dir.join("adapter_config.json"), cfg.to_string()).unwrap();
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (i, (module, in_f, out_f)) in modules.iter().enumerate() {
        let a = Tensor::from_vec(
            det_vals(rank * in_f, 1000.0 + i as f32),
            (rank, *in_f),
            &cpu,
        )
        .unwrap();
        let b = Tensor::from_vec(
            det_vals(out_f * rank, 2000.0 + i as f32),
            (*out_f, rank),
            &cpu,
        )
        .unwrap();
        tensors.insert(format!("{module}.lora_A.weight"), a);
        tensors.insert(format!("{module}.lora_B.weight"), b);
    }
    candle_core::safetensors::save(&tensors, dir.join("adapter_model.safetensors")).unwrap();
}

fn load_adapter(dir: &Path) -> LoraAdapter {
    LoraAdapter::load(dir, &Device::Cpu).unwrap()
}

fn bits_of(t: &Tensor) -> Vec<u32> {
    t.contiguous()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn probe_x(k: usize) -> Tensor {
    Tensor::from_vec(det_vals(3 * k, 5.5), (3usize, k), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

const H: usize = 32;
const Q_OUT: usize = 32;
const KV_OUT: usize = 16;
const G_OUT: usize = 4;
const INTER: usize = 64;

#[test]
fn install_maps_every_hookable_site_and_detach_restores_base_bitwise() {
    let _one = serialize();
    let ckpt = scratch("ck-full");
    write_tiny_checkpoint(&ckpt, true);
    let model = build_tiny_model(&ckpt, &Device::Cpu);

    let ada = scratch("ada-full");
    write_adapter(
        &ada,
        &[
            ("model.layers.0.self_attn.q_proj", H, Q_OUT),
            ("model.layers.0.self_attn.k_proj", H, KV_OUT),
            ("model.layers.0.self_attn.v_proj", H, KV_OUT),
            ("model.layers.0.self_attn.o_proj", Q_OUT, H),
            ("model.layers.0.self_attn.g_proj", H, G_OUT),
            ("model.layers.0.mlp.gate_proj", H, INTER),
            ("model.layers.0.mlp.up_proj", H, INTER),
            ("model.layers.0.mlp.down_proj", INTER, H),
            ("model.layers.1.self_attn.q_proj", H, Q_OUT),
        ],
        2,
    );
    let adapter = load_adapter(&ada);
    let prefix = LAGUNA_SERVING_LORA_PREFIX_IS_THE_GEMMA4_NAMESPACE_BECAUSE_NV_WEIGHTS_NORMALIZE_MODULE_NAME_REWRITES_MODEL_LAYERS;
    assert!(
        adapter
            .loras
            .contains_key(&format!("{prefix}.layers.0.self_attn.q_proj")),
        "the loader must rewrite model.layers.* into the {prefix} namespace; if this fails \
         the install map below is testing the wrong key space"
    );

    let l0 = &model.layers()[0];
    let q0 = &l0.self_attn.q_proj;
    let x = probe_x(H);
    let base_bits = bits_of(&q0.forward(&x).unwrap());

    let install = install_laguna_serving_adapter(&model, &adapter).expect("install");
    assert_eq!(install.matched_modules, 9);
    assert_eq!(install.hooked_linears, 9);
    assert!(install.skipped.is_empty(), "cpu model has no w8 to disable");
    assert!(q0.has_lora());
    assert!(l0.self_attn.k_proj.has_lora());
    assert!(l0.self_attn.v_proj.has_lora());
    assert!(l0.self_attn.o_proj.has_lora());
    assert!(l0.self_attn.g_proj.as_ref().unwrap().has_lora());
    let LagunaFfn::Dense(mlp) = &l0.ffn else {
        panic!("layer 0 of the tiny config must be dense")
    };
    assert!(mlp.gate_proj().has_lora());
    assert!(mlp.up_proj().has_lora());
    assert!(mlp.down_proj().has_lora());
    assert!(model.layers()[1].self_attn.q_proj.has_lora());

    let hooked_bits = bits_of(&q0.forward(&x).unwrap());
    assert_ne!(
        base_bits, hooked_bits,
        "an attached q_proj hook must change the forward output"
    );

    detach_laguna_serving_adapter(&model);
    assert!(!q0.has_lora());
    assert!(!l0.self_attn.g_proj.as_ref().unwrap().has_lora());
    assert!(!mlp.down_proj().has_lora());
    assert!(!model.layers()[1].self_attn.q_proj.has_lora());
    assert_eq!(
        base_bits,
        bits_of(&q0.forward(&x).unwrap()),
        "detach must restore the base forward bitwise"
    );

    let _ = std::fs::remove_dir_all(&ckpt);
    let _ = std::fs::remove_dir_all(&ada);
}

#[test]
fn moe_layer_mlp_target_refuses_the_whole_adapter_with_no_partial_attach() {
    let _one = serialize();
    let ckpt = scratch("ck-moe");
    write_tiny_checkpoint(&ckpt, true);
    let model = build_tiny_model(&ckpt, &Device::Cpu);

    for moe_module in [
        "model.layers.1.mlp.experts.0.gate_proj",
        "model.layers.1.mlp.shared_expert.down_proj",
        "model.layers.1.mlp.gate",
    ] {
        let ada = scratch("ada-moe");
        let moe_in = if moe_module.ends_with("down_proj") { 16 } else { H };
        let moe_out = match moe_module {
            m if m.ends_with("down_proj") => H,
            m if m.ends_with(".gate") => 2,
            _ => 16,
        };
        write_adapter(
            &ada,
            &[
                ("model.layers.0.self_attn.q_proj", H, Q_OUT),
                (moe_module, moe_in, moe_out),
            ],
            2,
        );
        let err = install_laguna_serving_adapter(&model, &load_adapter(&ada))
            .err()
            .unwrap_or_else(|| panic!("{moe_module} must refuse"));
        let msg = format!("{err:#}");
        assert!(
            msg.contains("MoE") && msg.contains("grouped expert kernels"),
            "refusal must name the grouped-kernel bypass, got: {msg}"
        );
        assert!(
            !model.layers()[0].self_attn.q_proj.has_lora(),
            "a refused adapter must leave no partial application"
        );
        let _ = std::fs::remove_dir_all(&ada);
    }
    let _ = std::fs::remove_dir_all(&ckpt);
}

#[test]
fn unmatched_module_beside_matched_ones_refuses_naming_the_subset_hazard() {
    let _one = serialize();
    let ckpt = scratch("ck-unm");
    write_tiny_checkpoint(&ckpt, true);
    let model = build_tiny_model(&ckpt, &Device::Cpu);
    let ada = scratch("ada-unm");
    write_adapter(
        &ada,
        &[
            ("model.layers.0.self_attn.q_proj", H, Q_OUT),
            ("model.layers.0.self_attn.qkv_proj", H, Q_OUT),
        ],
        2,
    );
    let err = install_laguna_serving_adapter(&model, &load_adapter(&ada))
        .err()
        .expect("an unmatched module beside matched ones must refuse");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("module subset"),
        "refusal must name the partial-serving hazard, got: {msg}"
    );
    assert!(!model.layers()[0].self_attn.q_proj.has_lora());
    let _ = std::fs::remove_dir_all(&ckpt);
    let _ = std::fs::remove_dir_all(&ada);
}

#[test]
fn zero_matches_refuses_naming_the_normalized_namespace() {
    let _one = serialize();
    let ckpt = scratch("ck-zero");
    write_tiny_checkpoint(&ckpt, true);
    let model = build_tiny_model(&ckpt, &Device::Cpu);
    let ada = scratch("ada-zero");
    write_adapter(&ada, &[("transformer.h.0.attn.c_attn", H, Q_OUT)], 2);
    let err = install_laguna_serving_adapter(&model, &load_adapter(&ada))
        .err()
        .expect("an off-family adapter must refuse");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("normalize_module_name"),
        "the zero-match refusal must tell the caller which key space install expects, got: {msg}"
    );
    let _ = std::fs::remove_dir_all(&ckpt);
    let _ = std::fs::remove_dir_all(&ada);
}

#[test]
fn g_proj_target_on_an_ungated_config_refuses() {
    let _one = serialize();
    let ckpt = scratch("ck-nog");
    write_tiny_checkpoint(&ckpt, false);
    let model = build_tiny_model(&ckpt, &Device::Cpu);
    assert!(model.layers()[0].self_attn.g_proj.is_none());
    let ada = scratch("ada-nog");
    write_adapter(&ada, &[("model.layers.0.self_attn.g_proj", H, G_OUT)], 2);
    let err = install_laguna_serving_adapter(&model, &load_adapter(&ada))
        .err()
        .expect("a g_proj target must refuse when the config declares gating=none");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("gating=none"),
        "refusal must name the structural absence, got: {msg}"
    );
    let _ = std::fs::remove_dir_all(&ckpt);
    let _ = std::fs::remove_dir_all(&ada);
}

#[cfg(feature = "cuda")]
mod cuda_w8_engagement {
    use super::*;
    use nv_specdecode::lora_spec::serving_hook_engagement_counted_host_side_so_replays_of_a_captured_graph_leave_it_flat as engagement;

    fn load_adapter_on_the_model_device_so_the_cuda_m1_kernel_delta_actually_builds(
        dir: &Path,
        device: &Device,
    ) -> LoraAdapter {
        LoraAdapter::load(dir, device).unwrap()
    }

    fn probe_x_m1_bf16(device: &Device) -> Tensor {
        Tensor::from_vec(det_vals(H, 5.5), (1usize, H), device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }

    #[test]
    fn install_zeroes_the_w8_scope_so_hooked_qkvo_take_linear_forward_and_detach_rearms() {
        let _one = serialize();
        let ckpt = scratch("ck-w8");
        write_tiny_checkpoint(&ckpt, true);
        let device = Device::new_cuda(0).expect("cuda device");
        let model = build_tiny_model(&ckpt, &device);
        assert!(
            model.attn_w8_active(),
            "the fp8 attn projections are the round-82 default and must build on this tiny \
             bf16 shape; if this fails the suite's w8-bypass coverage went vacuous"
        );
        let boot_scope = model.attn_w8_scope();
        assert_ne!(boot_scope, 0);

        let q0 = &model.layers()[0].self_attn.q_proj;
        let x1 = probe_x_m1_bf16(&device);
        let base_bits = bits_of(&q0.forward(&x1).unwrap());

        let ada = scratch("ada-w8");
        write_adapter(&ada, &[("model.layers.0.self_attn.q_proj", H, Q_OUT)], 2);
        let adapter =
            load_adapter_on_the_model_device_so_the_cuda_m1_kernel_delta_actually_builds(
                &ada, &device,
            );
        let install = install_laguna_serving_adapter(&model, &adapter).expect("install");
        assert_eq!(
            model.attn_w8_scope(),
            0,
            "install must zero the w8 scope: the quantized gemv bypasses Linear::forward, so \
             a nonzero scope means the attached q_proj hook never fires at m=1"
        );
        assert!(
            install.skipped.iter().any(|s| s.contains("w8")),
            "the install report must say the w8 projections were disabled"
        );

        let eng_before = engagement();
        let hooked_bits = bits_of(&q0.forward(&x1).unwrap());
        let eng_after = engagement();
        assert_ne!(
            base_bits, hooked_bits,
            "an m=1 bf16 cuda forward through the hooked q_proj must change the output; \
             equality means the hook never fired"
        );
        assert!(
            eng_after.m1_kernel_applies > eng_before.m1_kernel_applies,
            "the m=1 bf16 cuda forward must engage the nv-kernels two-phase kernel, not the \
             candle fallback; a flat counter means the adapter slices were not on the cuda \
             device and build_cuda_m1_kernel_delta returned None"
        );
        assert_eq!(
            eng_after.fallback_candle_applies, eng_before.fallback_candle_applies,
            "the candle fallback must stay flat when the m=1 kernel path is available"
        );

        detach_laguna_serving_adapter(&model);
        assert_eq!(
            model.attn_w8_scope(),
            boot_scope,
            "detach must re-arm the boot w8 scope"
        );
        assert_eq!(
            base_bits,
            bits_of(&q0.forward(&x1).unwrap()),
            "detach must restore the base m=1 forward bitwise"
        );
        let _ = std::fs::remove_dir_all(&ckpt);
        let _ = std::fs::remove_dir_all(&ada);
    }

    #[test]
    fn mlp_only_adapter_leaves_the_w8_projections_armed() {
        let _one = serialize();
        let ckpt = scratch("ck-w8mlp");
        write_tiny_checkpoint(&ckpt, true);
        let device = Device::new_cuda(0).expect("cuda device");
        let model = build_tiny_model(&ckpt, &device);
        assert!(model.attn_w8_active());
        let boot_scope = model.attn_w8_scope();
        let ada = scratch("ada-w8mlp");
        write_adapter(&ada, &[("model.layers.0.mlp.down_proj", INTER, H)], 2);
        let adapter =
            load_adapter_on_the_model_device_so_the_cuda_m1_kernel_delta_actually_builds(
                &ada, &device,
            );
        install_laguna_serving_adapter(&model, &adapter).expect("install");
        assert_eq!(
            model.attn_w8_scope(),
            boot_scope,
            "an adapter that hooks no q/k/v/o site must not pay the w8 disable"
        );
        detach_laguna_serving_adapter(&model);
        let _ = std::fs::remove_dir_all(&ckpt);
        let _ = std::fs::remove_dir_all(&ada);
    }

    struct ShapePolicyEnvRestoredOnDropSoAPanicCannotLeakIntoLaterModelBuilds(Option<String>);

    impl Drop for ShapePolicyEnvRestoredOnDropSoAPanicCannotLeakIntoLaterModelBuilds {
        fn drop(&mut self) {
            match self.0.take() {
                Some(v) => std::env::set_var("NV_LAGUNA_ATTN_FP8", v),
                None => std::env::remove_var("NV_LAGUNA_ATTN_FP8"),
            }
        }
    }

    #[test]
    fn shape_policy_boot_refuses_attention_adapters_because_the_disable_cannot_stick() {
        let _one = serialize();
        let _restore = ShapePolicyEnvRestoredOnDropSoAPanicCannotLeakIntoLaterModelBuilds(
            std::env::var("NV_LAGUNA_ATTN_FP8").ok(),
        );
        std::env::set_var("NV_LAGUNA_ATTN_FP8", "shape:qkvo");
        let ckpt = scratch("ck-shape");
        write_tiny_checkpoint(&ckpt, true);
        let device = Device::new_cuda(0).expect("cuda device");
        let model = build_tiny_model(&ckpt, &device);
        assert!(model.attn_w8_shape_policy());
        assert!(model.attn_w8_active());
        let ada = scratch("ada-shape");
        write_adapter(&ada, &[("model.layers.0.self_attn.q_proj", H, Q_OUT)], 2);
        let adapter =
            load_adapter_on_the_model_device_so_the_cuda_m1_kernel_delta_actually_builds(
                &ada, &device,
            );
        let err = install_laguna_serving_adapter(&model, &adapter)
            .err()
            .expect("the shape policy re-arms the w8 gemv every m=1 forward, so install must refuse");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("shape"),
            "refusal must name the shape policy, got: {msg}"
        );
        assert!(!model.layers()[0].self_attn.q_proj.has_lora());
        let _ = std::fs::remove_dir_all(&ckpt);
        let _ = std::fs::remove_dir_all(&ada);
    }
}
