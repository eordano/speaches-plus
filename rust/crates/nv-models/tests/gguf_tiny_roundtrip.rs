mod hub_snapshot;

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4_gguf::{gemma4_moe_config_from_gguf, gemma4_moe_config_json_from_gguf};
use nv_models::gemma4_moe::Gemma4Moe;
use nv_weights::GgufLoader;
mod common;
use common::TempDir;

const HIDDEN: usize = 64;
const INTER: usize = 96;
const N_LAYERS: usize = 3;
const N_Q: usize = 4;
const N_KV: usize = 2;
const N_GLOBAL_KV: usize = 1;
const HEAD_DIM: usize = 16;
const GLOBAL_HEAD_DIM: usize = 32;
const VOCAB: usize = 160;
const N_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const MOE_INTER: usize = 64;
const WINDOW: usize = 8;
const EOS_ID: u32 = 5;

fn temp_dir(tag: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("gguf_tiny_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

fn lcg_tensor(seed: &mut u64, shape: &[usize], scale: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((*seed >> 33) as u32) as f32 / (u32::MAX as f32);
        v.push((u * 2.0 - 1.0) * scale);
    }
    Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
}

fn ones(shape: &[usize]) -> Tensor {
    Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
}

fn write_tiny_gguf(dir: &std::path::Path) -> std::path::PathBuf {
    let mut seed = 0x5eed_cafe_u64;
    let q4 = |t: &Tensor| QTensor::quantize(t, GgmlDType::Q4_0).unwrap();
    let f32q = |t: &Tensor| QTensor::quantize(t, GgmlDType::F32).unwrap();

    let mut tensors: Vec<(String, QTensor)> = Vec::new();
    tensors.push((
        "token_embd.weight".into(),
        q4(&lcg_tensor(&mut seed, &[VOCAB, HIDDEN], 0.05)),
    ));
    tensors.push(("output_norm.weight".into(), f32q(&ones(&[HIDDEN]))));
    for i in 0..N_LAYERS {
        let global = i == N_LAYERS - 1;
        let (n_kv, hd) = if global {
            (N_GLOBAL_KV, GLOBAL_HEAD_DIM)
        } else {
            (N_KV, HEAD_DIM)
        };
        let p = |s: &str| format!("blk.{i}.{s}");
        for norm in [
            "attn_norm.weight",
            "post_attention_norm.weight",
            "ffn_norm.weight",
            "post_ffw_norm.weight",
            "post_ffw_norm_1.weight",
            "pre_ffw_norm_2.weight",
            "post_ffw_norm_2.weight",
        ] {
            tensors.push((p(norm), f32q(&ones(&[HIDDEN]))));
        }
        tensors.push((p("attn_q_norm.weight"), f32q(&ones(&[hd]))));
        tensors.push((p("attn_k_norm.weight"), f32q(&ones(&[hd]))));
        tensors.push((
            p("attn_q.weight"),
            q4(&lcg_tensor(&mut seed, &[N_Q * hd, HIDDEN], 0.05)),
        ));
        tensors.push((
            p("attn_k.weight"),
            q4(&lcg_tensor(&mut seed, &[n_kv * hd, HIDDEN], 0.05)),
        ));
        if !global {
            tensors.push((
                p("attn_v.weight"),
                q4(&lcg_tensor(&mut seed, &[n_kv * hd, HIDDEN], 0.05)),
            ));
        }
        tensors.push((
            p("attn_output.weight"),
            q4(&lcg_tensor(&mut seed, &[HIDDEN, N_Q * hd], 0.05)),
        ));
        tensors.push((
            p("ffn_gate.weight"),
            q4(&lcg_tensor(&mut seed, &[INTER, HIDDEN], 0.05)),
        ));
        tensors.push((
            p("ffn_up.weight"),
            q4(&lcg_tensor(&mut seed, &[INTER, HIDDEN], 0.05)),
        ));
        tensors.push((
            p("ffn_down.weight"),
            q4(&lcg_tensor(&mut seed, &[HIDDEN, INTER], 0.05)),
        ));
        tensors.push((
            p("ffn_gate_inp.weight"),
            q4(&lcg_tensor(&mut seed, &[N_EXPERTS, HIDDEN], 0.05)),
        ));
        tensors.push((p("ffn_gate_inp.scale"), f32q(&ones(&[HIDDEN]))));
        tensors.push((p("ffn_down_exps.scale"), f32q(&ones(&[N_EXPERTS]))));
        tensors.push((
            p("ffn_gate_up_exps.weight"),
            q4(&lcg_tensor(
                &mut seed,
                &[N_EXPERTS, 2 * MOE_INTER, HIDDEN],
                0.05,
            )),
        ));
        tensors.push((
            p("ffn_down_exps.weight"),
            q4(&lcg_tensor(
                &mut seed,
                &[N_EXPERTS, HIDDEN, MOE_INTER],
                0.05,
            )),
        ));
        tensors.push((p("layer_output_scale.weight"), f32q(&ones(&[1]))));
    }

    let pattern: Vec<Value> = (0..N_LAYERS)
        .map(|i| Value::Bool(i != N_LAYERS - 1))
        .collect();
    let kv_list: Vec<Value> = (0..N_LAYERS)
        .map(|i| {
            Value::U32(if i == N_LAYERS - 1 {
                N_GLOBAL_KV as u32
            } else {
                N_KV as u32
            })
        })
        .collect();
    let metadata: Vec<(&str, Value)> = vec![
        ("general.architecture", Value::String("gemma4".into())),
        ("gemma4.block_count", Value::U32(N_LAYERS as u32)),
        ("gemma4.embedding_length", Value::U32(HIDDEN as u32)),
        ("gemma4.feed_forward_length", Value::U32(INTER as u32)),
        ("gemma4.attention.head_count", Value::U32(N_Q as u32)),
        ("gemma4.attention.head_count_kv", Value::Array(kv_list)),
        (
            "gemma4.attention.key_length_swa",
            Value::U32(HEAD_DIM as u32),
        ),
        (
            "gemma4.attention.key_length",
            Value::U32(GLOBAL_HEAD_DIM as u32),
        ),
        ("gemma4.context_length", Value::U32(64)),
        ("gemma4.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        ("gemma4.attention.sliding_window", Value::U32(WINDOW as u32)),
        ("gemma4.final_logit_softcapping", Value::F32(30.0)),
        ("gemma4.rope.freq_base", Value::F32(1_000_000.0)),
        ("gemma4.rope.freq_base_swa", Value::F32(10_000.0)),
        (
            "gemma4.rope.dimension_count",
            Value::U32((GLOBAL_HEAD_DIM / 4) as u32),
        ),
        (
            "gemma4.attention.sliding_window_pattern",
            Value::Array(pattern),
        ),
        ("gemma4.expert_count", Value::U32(N_EXPERTS as u32)),
        ("gemma4.expert_used_count", Value::U32(TOP_K as u32)),
        (
            "gemma4.expert_feed_forward_length",
            Value::U32(MOE_INTER as u32),
        ),
        ("tokenizer.ggml.eos_token_id", Value::U32(EOS_ID)),
    ];

    let path = dir.join("model.gguf");
    let mut f = std::fs::File::create(&path).unwrap();
    let md_refs: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let t_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    gguf_file::write(&mut f, &md_refs, &t_refs).unwrap();
    path
}

#[test]
fn tiny_gguf_config_eos_and_cpu_decode() {
    let dir = temp_dir("cpu");
    let path = write_tiny_gguf(&dir.0);

    assert_eq!(
        nv_weights::gguf::lone_gguf_file(&dir.0).as_deref(),
        Some(path.as_path()),
        "the dir resolves as a GGUF checkpoint dir"
    );

    let loader = GgufLoader::open(&path, &Device::Cpu).unwrap();
    let json = gemma4_moe_config_json_from_gguf(&loader).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["model_type"], "gemma4");
    assert_eq!(v["enable_moe_block"], true);
    assert_eq!(v["num_experts"], N_EXPERTS);
    assert_eq!(v["vocab_size"], VOCAB);
    assert_eq!(v["tie_word_embeddings"], true);
    assert_eq!(
        v["layer_types"].as_array().unwrap().len(),
        N_LAYERS,
        "sliding_window_pattern -> layer_types"
    );

    assert_eq!(
        loader.md_u64("tokenizer.ggml.eos_token_id").unwrap(),
        EOS_ID as u64,
        "eos lives in gguf metadata, no sidecar needed"
    );

    let cfg = gemma4_moe_config_from_gguf(&loader).unwrap();
    assert_eq!(cfg.base.num_hidden_layers, N_LAYERS);

    let model = Gemma4Moe::from_gguf(&path, &Device::Cpu, DType::BF16).unwrap();
    let mut cache = model.new_kv_cache(16).unwrap();
    for (i, t) in [2u32, 7, 11].iter().enumerate() {
        let tok = Tensor::from_vec(vec![*t], (1usize, 1usize), &Device::Cpu).unwrap();
        let pos = Tensor::from_vec(vec![i as i32], 1usize, &Device::Cpu).unwrap();
        let logits: Vec<f32> = model
            .forward_with_cache(&tok, &pos, &mut cache)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(logits.len(), VOCAB);
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "step {i}: non-finite logits from the gguf-built model"
        );
    }
}

#[cfg(feature = "wgpu")]
mod wgpu_roundtrip {
    use super::*;
    use nv_models::gemma4_moe_wgpu::{dequantize_w4_expert, host_layer_from_loader, Gemma4MoeWgpu};
    use nv_weights::{TensorSource, WeightLoader};
    use std::collections::HashMap;

    fn have_gpu() -> bool {
        match nv_kernels::wgpu_backend::WgpuContext::shared() {
            Ok(ctx) => {
                eprintln!("[wgpu] adapter: {}", ctx.info.name);
                true
            }
            Err(e) => {
                super::hub_snapshot::precondition_absent(
                    "tiny_gguf_wgpu_matches_dequant_oracle",
                    &format!("no wgpu adapter: {e}"),
                    "run under rust/scripts/nvk.sh, which wires VK_ICD_FILENAMES and prepends \
                     the store vulkan-loader to LD_LIBRARY_PATH",
                );
                false
            }
        }
    }

    fn rel_err(a: &[f32], b: &[f32]) -> (f32, f32) {
        let mut maxabs = 0f32;
        let mut scale = 0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            maxabs = maxabs.max((x - y).abs());
            scale = scale.max(x.abs()).max(y.abs());
        }
        (maxabs, maxabs / scale.max(1e-6))
    }

    fn argmax(v: &[f32]) -> u32 {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap()
    }

    #[test]
    fn tiny_gguf_wgpu_matches_dequant_oracle() {
        if !have_gpu() {
            return;
        }
        let dir = temp_dir("wgpu");
        let path = write_tiny_gguf(&dir.0);

        let gguf = GgufLoader::open(&path, &Device::Cpu).unwrap();
        let cfg = gemma4_moe_config_from_gguf(&gguf).unwrap();
        let mut gpu = Gemma4MoeWgpu::from_gguf(&path, 16).expect("build wgpu engine from gguf");

        let st = dir.0.join("oracle.safetensors");
        {
            let mut out: HashMap<String, Tensor> = HashMap::new();
            for name in [
                "model.language_model.embed_tokens.weight",
                "model.language_model.norm.weight",
            ] {
                out.insert(name.to_string(), gguf.get(name, DType::BF16).unwrap());
            }
            for i in 0..N_LAYERS {
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
                    out.insert(name.clone(), gguf.get(&name, DType::BF16).unwrap());
                }
                let vname = format!("{p}.self_attn.v_proj.weight");
                if gguf.has(&vname) {
                    out.insert(vname.clone(), gguf.get(&vname, DType::BF16).unwrap());
                }
                let layer = host_layer_from_loader(&cfg, &gguf, i).unwrap();
                let (hidden, mi, n_e) = (HIDDEN, MOE_INTER, N_EXPERTS);
                let mut gu = vec![0f32; n_e * 2 * mi * hidden];
                let mut dn = vec![0f32; n_e * hidden * mi];
                for e in 0..n_e {
                    let gate = dequantize_w4_expert(&layer.experts_gate, e);
                    let up = dequantize_w4_expert(&layer.experts_up, e);
                    let down = dequantize_w4_expert(&layer.experts_down, e);
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
            candle_core::safetensors::save(&out, &st).unwrap();
        }
        let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();

        let oracle =
            Gemma4Moe::from_loader_dtype(cfg.clone(), &loader, &Device::Cpu, DType::BF16).unwrap();
        let mut cache = oracle.new_kv_cache(16).unwrap();

        let mut worst_rel = 0f32;
        for (i, t) in [2u32, 7, 11, 3].iter().enumerate() {
            let (arg, logits) = gpu.decode_step_logits(*t).expect("decode step");
            let tok = Tensor::from_vec(vec![*t], (1usize, 1usize), &Device::Cpu).unwrap();
            let pos = Tensor::from_vec(vec![i as i32], 1usize, &Device::Cpu).unwrap();
            let want: Vec<f32> = oracle
                .forward_with_cache(&tok, &pos, &mut cache)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let (abs, rel) = rel_err(&logits, &want);
            let ref_arg = argmax(&want);
            eprintln!(
                "[tiny-gguf-wgpu] step {i}: gpu_argmax={arg} ref_argmax={ref_arg} \
                 max_abs={abs:.6} rel={rel:.6}"
            );
            worst_rel = worst_rel.max(rel);
            assert!(
                rel < 0.05,
                "step {i}: wgpu diverged from oracle (rel {rel})"
            );
            if arg != ref_arg {
                let gap = want[ref_arg as usize] - want[arg as usize];
                assert!(
                    (0.0..=abs.max(1e-3) * 2.0).contains(&gap),
                    "step {i}: argmax diverged beyond numerical tie (gap {gap}, abs {abs})"
                );
            }
        }
        eprintln!("[tiny-gguf-wgpu] worst rel over 4 steps: {worst_rel:.6}");
    }
}
