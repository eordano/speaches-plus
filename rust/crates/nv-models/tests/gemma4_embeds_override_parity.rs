use candle_core::{DType, Device, Tensor};
mod common;
use common::LcgTop24TwoSided as Lcg;
use common::TempDir;

fn rand_tensor(rng: &mut Lcg, shape: &[usize], scale: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32() * scale).collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

fn norm_tensor(rng: &mut Lcg, dim: usize) -> Tensor {
    let data: Vec<f32> = (0..dim).map(|_| 1.0 + 0.25 * rng.next_f32()).collect();
    Tensor::from_vec(data, dim, &Device::Cpu).unwrap()
}

fn temp_dir(tag: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("g4_embeds_parity_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

fn embed_rows_as_mm_embeddings_does(
    embed_weight: &Tensor,
    embed_scale: f64,
    ids: &[u32],
    device: &Device,
) -> Tensor {
    let idx = Tensor::from_vec(ids.to_vec(), ids.len(), device).unwrap();
    embed_weight
        .index_select(&idx, 0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .affine(embed_scale, 0.0)
        .unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    let v: Vec<f32> = t
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    v.iter().map(|x| x.to_bits()).collect()
}

mod moe {
    use super::*;
    use nv_models::gemma4::LayerType;
    use nv_models::gemma4_moe::{Gemma4Moe, Gemma4MoeConfig};
    use nv_weights::WeightLoader;
    use std::collections::HashMap;

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

    fn tiny_config_json() -> String {
        format!(
            r#"{{
  "model_type": "gemma4",
  "tie_word_embeddings": true,
  "text_config": {{
    "attention_k_eq_v": true,
    "enable_moe_block": true,
    "final_logit_softcapping": 30.0,
    "global_head_dim": {GLOBAL_HEAD_DIM},
    "head_dim": {HEAD_DIM},
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": {HIDDEN},
    "intermediate_size": {INTER},
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
    "max_position_embeddings": 64,
    "moe_intermediate_size": {MOE_INTER},
    "num_attention_heads": {N_Q},
    "num_experts": {N_EXPERTS},
    "num_global_key_value_heads": {N_GLOBAL_KV},
    "num_hidden_layers": {N_LAYERS},
    "num_key_value_heads": {N_KV},
    "rms_norm_eps": 1e-06,
    "rope_parameters": {{
      "full_attention": {{"partial_rotary_factor": 0.25, "rope_theta": 1000000.0}},
      "sliding_attention": {{"rope_theta": 10000.0}}
    }},
    "sliding_window": {WINDOW},
    "tie_word_embeddings": true,
    "top_k_experts": {TOP_K},
    "vocab_size": {VOCAB}
  }}
}}"#
        )
    }

    fn tensors_for_cfg(cfg: &Gemma4MoeConfig, seed: u64) -> HashMap<String, Tensor> {
        let base = &cfg.base;
        let (hidden, inter, vocab) = (base.hidden_size, base.intermediate_size, base.vocab_size);
        let (n_q, n_e, mi) = (
            base.num_attention_heads,
            cfg.num_experts,
            cfg.moe_intermediate_size,
        );
        let mut rng = Lcg(seed);
        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert(
            "model.language_model.embed_tokens.weight".into(),
            rand_tensor(&mut rng, &[vocab, hidden], 1.0),
        );
        t.insert(
            "model.language_model.norm.weight".into(),
            norm_tensor(&mut rng, hidden),
        );
        for i in 0..base.num_hidden_layers {
            let p = format!("model.language_model.layers.{i}");
            let kind = base.layer_kind(i);
            let full = kind == LayerType::FullAttention;
            let hd = base.head_dim_for(kind);
            let n_kv = base.num_kv_heads_for(kind);
            for norm in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "post_feedforward_layernorm_1",
                "pre_feedforward_layernorm_2",
                "post_feedforward_layernorm_2",
            ] {
                t.insert(format!("{p}.{norm}.weight"), norm_tensor(&mut rng, hidden));
            }
            t.insert(
                format!("{p}.layer_scalar"),
                Tensor::from_vec(vec![0.9f32 + 0.1 * rng.next_f32()], 1, &Device::Cpu).unwrap(),
            );
            t.insert(
                format!("{p}.self_attn.q_proj.weight"),
                rand_tensor(&mut rng, &[n_q * hd, hidden], 0.3),
            );
            t.insert(
                format!("{p}.self_attn.k_proj.weight"),
                rand_tensor(&mut rng, &[n_kv * hd, hidden], 0.3),
            );
            if !(full && base.attention_k_eq_v) {
                t.insert(
                    format!("{p}.self_attn.v_proj.weight"),
                    rand_tensor(&mut rng, &[n_kv * hd, hidden], 0.3),
                );
            }
            t.insert(
                format!("{p}.self_attn.o_proj.weight"),
                rand_tensor(&mut rng, &[hidden, n_q * hd], 0.3),
            );
            t.insert(
                format!("{p}.self_attn.q_norm.weight"),
                norm_tensor(&mut rng, hd),
            );
            t.insert(
                format!("{p}.self_attn.k_norm.weight"),
                norm_tensor(&mut rng, hd),
            );
            t.insert(
                format!("{p}.mlp.gate_proj.weight"),
                rand_tensor(&mut rng, &[inter, hidden], 0.3),
            );
            t.insert(
                format!("{p}.mlp.up_proj.weight"),
                rand_tensor(&mut rng, &[inter, hidden], 0.3),
            );
            t.insert(
                format!("{p}.mlp.down_proj.weight"),
                rand_tensor(&mut rng, &[hidden, inter], 0.3),
            );
            t.insert(
                format!("{p}.router.proj.weight"),
                rand_tensor(&mut rng, &[n_e, hidden], 0.3),
            );
            t.insert(format!("{p}.router.scale"), norm_tensor(&mut rng, hidden));
            t.insert(
                format!("{p}.router.per_expert_scale"),
                norm_tensor(&mut rng, n_e),
            );
            t.insert(
                format!("{p}.experts.gate_up_proj"),
                rand_tensor(&mut rng, &[n_e, 2 * mi, hidden], 0.3),
            );
            t.insert(
                format!("{p}.experts.down_proj"),
                rand_tensor(&mut rng, &[n_e, hidden, mi], 0.3),
            );
        }
        t
    }

    fn assert_affine_scale_survives_bf16(embed_scale: f32) {
        assert_eq!(
            embed_scale,
            f32::from(half::bf16::from_f32(embed_scale)),
            "Gemma4Moe's token path scales with Tensor::affine, which rounds the multiplier into \
             the tensor dtype, while mm_embeddings scales in f32; bit-identity is only claimable \
             for a hidden_size whose sqrt survives bf16, and HIDDEN={HIDDEN} was chosen for that"
        );
    }

    fn tiny_moe(dir: &std::path::Path) -> Gemma4Moe {
        let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
        let st = dir.join("model.safetensors");
        candle_core::safetensors::save(&tensors_for_cfg(&cfg, 0x5eed_cafe_f00d_0002), &st).unwrap();
        let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
        Gemma4Moe::from_loader_dtype(cfg, &loader, &Device::Cpu, DType::BF16).unwrap()
    }

    fn run_tokens(model: &Gemma4Moe, ids: &[u32], chunk: usize) -> Vec<u32> {
        let mut cache = model.new_kv_cache(64).unwrap();
        let mut last = None;
        let mut off = 0usize;
        while off < ids.len() {
            let c = chunk.min(ids.len() - off);
            let tok =
                Tensor::from_vec(ids[off..off + c].to_vec(), (1usize, c), &Device::Cpu).unwrap();
            let pos: Vec<i32> = (off..off + c).map(|p| p as i32).collect();
            let pos = Tensor::from_vec(pos, c, &Device::Cpu).unwrap();
            last = Some(model.forward_with_cache(&tok, &pos, &mut cache).unwrap());
            off += c;
        }
        bits(&last.unwrap())
    }

    fn run_embeds(model: &Gemma4Moe, ids: &[u32], embeds: &Tensor, chunk: usize) -> Vec<u32> {
        let mut cache = model.new_kv_cache(64).unwrap();
        let mut last = None;
        let mut off = 0usize;
        while off < ids.len() {
            let c = chunk.min(ids.len() - off);
            let tok =
                Tensor::from_vec(ids[off..off + c].to_vec(), (1usize, c), &Device::Cpu).unwrap();
            let pos: Vec<i32> = (off..off + c).map(|p| p as i32).collect();
            let pos = Tensor::from_vec(pos, c, &Device::Cpu).unwrap();
            let e = embeds.narrow(0, off, c).unwrap();
            last = Some(
                model
                    .forward_with_cache_embeds(&tok, &e, &pos, &mut cache)
                    .unwrap(),
            );
            off += c;
        }
        bits(&last.unwrap())
    }

    #[test]
    fn moe_embeds_override_reproduces_the_token_path_bit_for_bit() {
        let dir = temp_dir("moe");
        let model = tiny_moe(&dir.0);
        assert_affine_scale_survives_bf16(model.embed_scale());

        let ids: Vec<u32> = vec![3, 17, 42, 8, 91, 5, 60, 12];
        let embeds = embed_rows_as_mm_embeddings_does(
            model.embed_weight(),
            model.embed_scale() as f64,
            &ids,
            &Device::Cpu,
        );
        assert_eq!(embeds.dims(), [ids.len(), HIDDEN]);
        assert_eq!(embeds.dtype(), DType::F32);

        let want = run_tokens(&model, &ids, ids.len());
        assert_eq!(want.len(), ids.len() * VOCAB);
        assert_eq!(
            run_embeds(&model, &ids, &embeds, ids.len()),
            want,
            "one-shot embeds prefill diverged from the token path"
        );
        assert_eq!(
            run_embeds(&model, &ids, &embeds, 3),
            run_tokens(&model, &ids, 3),
            "chunk-narrowed embeds prefill diverged from the same-chunking token path"
        );
    }

    #[test]
    fn moe_embeds_override_is_actually_read() {
        let dir = temp_dir("moe_perturb");
        let model = tiny_moe(&dir.0);
        let ids: Vec<u32> = vec![3, 17, 42, 8];
        let embeds = embed_rows_as_mm_embeddings_does(
            model.embed_weight(),
            model.embed_scale() as f64,
            &ids,
            &Device::Cpu,
        );
        let base = run_embeds(&model, &ids, &embeds, ids.len());

        let mut rows: Vec<f32> = embeds.flatten_all().unwrap().to_vec1().unwrap();
        for x in rows.iter_mut().skip(HIDDEN).take(HIDDEN) {
            *x += 1.0;
        }
        let perturbed = Tensor::from_vec(rows, (ids.len(), HIDDEN), &Device::Cpu).unwrap();
        assert_ne!(
            run_embeds(&model, &ids, &perturbed, ids.len()),
            base,
            "perturbing one embed row left the logits untouched: the override is being ignored"
        );

        let mut short = model.new_kv_cache(64).unwrap();
        let tok = Tensor::from_vec(ids.clone(), (1usize, ids.len()), &Device::Cpu).unwrap();
        let pos: Vec<i32> = (0..ids.len() as i32).collect();
        let pos = Tensor::from_vec(pos, ids.len(), &Device::Cpu).unwrap();
        let wrong = embeds.narrow(0, 0, ids.len() - 1).unwrap();
        assert!(
            model
                .forward_with_cache_embeds(&tok, &wrong, &pos, &mut short)
                .is_err(),
            "a [seq-1, hidden] override must be rejected, not silently broadcast"
        );
    }
}

#[cfg(feature = "cuda")]
mod dense {
    use super::*;
    use nv_models::gemma4::{Gemma4, Gemma4Config};
    use nv_weights::WeightLoader;
    use std::collections::HashMap;

    const HIDDEN: usize = 128;
    const INTER: usize = 256;
    const N_LAYERS: usize = 2;
    const N_Q: usize = 2;
    const N_KV: usize = 1;
    const HEAD_DIM: usize = 128;
    const VOCAB: usize = 512;

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
  "sliding_window": 32,
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

    fn tiny_dense(dir: &std::path::Path, device: &Device) -> Gemma4 {
        let mut rng = Lcg(0x5eed_cafe_f00d_0003);
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let bf16 = |x: Tensor| x.to_dtype(DType::BF16).unwrap();
        t.insert(
            "model.language_model.embed_tokens.weight".into(),
            bf16(rand_tensor(&mut rng, &[VOCAB, HIDDEN], 1.0)),
        );
        t.insert(
            "model.language_model.norm.weight".into(),
            bf16(norm_tensor(&mut rng, HIDDEN)),
        );
        t.insert(
            "lm_head.weight".into(),
            bf16(rand_tensor(&mut rng, &[VOCAB, HIDDEN], 1.0)),
        );
        for i in 0..N_LAYERS {
            let p = format!("model.language_model.layers.{i}");
            for norm in [
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ] {
                t.insert(
                    format!("{p}.{norm}.weight"),
                    bf16(norm_tensor(&mut rng, HIDDEN)),
                );
            }
            t.insert(
                format!("{p}.layer_scalar"),
                bf16(Tensor::ones(1, DType::F32, &Device::Cpu).unwrap()),
            );
            for (name, shape) in [
                ("self_attn.q_proj.weight", [N_Q * HEAD_DIM, HIDDEN]),
                ("self_attn.k_proj.weight", [N_KV * HEAD_DIM, HIDDEN]),
                ("self_attn.v_proj.weight", [N_KV * HEAD_DIM, HIDDEN]),
                ("self_attn.o_proj.weight", [HIDDEN, N_Q * HEAD_DIM]),
                ("mlp.gate_proj.weight", [INTER, HIDDEN]),
                ("mlp.up_proj.weight", [INTER, HIDDEN]),
                ("mlp.down_proj.weight", [HIDDEN, INTER]),
            ] {
                t.insert(
                    format!("{p}.{name}"),
                    bf16(rand_tensor(&mut rng, &shape, 0.3)),
                );
            }
            for name in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
                t.insert(format!("{p}.{name}"), bf16(norm_tensor(&mut rng, HEAD_DIM)));
            }
        }
        let st = dir.join("model.safetensors");
        candle_core::safetensors::save(&t, &st).unwrap();
        let cfg = Gemma4Config::from_hf_json_str(&config_json()).unwrap();
        let weights = WeightLoader::open_file(&st, device).unwrap();
        Gemma4::from_loader(cfg, &weights, device).unwrap()
    }

    fn cuda() -> Device {
        Device::new_cuda(0).expect(
            "gemma4_embeds_override_parity dense arm needs CUDA device 0: with feature=cuda the \
             dense embed path is embed_lookup_bf16_op, which has no CPU branch. This gate refuses \
             to report success without running.",
        )
    }

    fn run_tokens(model: &Gemma4, device: &Device, ids: &[u32], chunk: usize) -> Vec<u32> {
        let mut cache = model.new_kv_cache(64).unwrap();
        let mut last = None;
        let mut off = 0usize;
        while off < ids.len() {
            let c = chunk.min(ids.len() - off);
            let tok = Tensor::from_vec(ids[off..off + c].to_vec(), (1usize, c), device).unwrap();
            let pos: Vec<i32> = (off..off + c).map(|p| p as i32).collect();
            let pos = Tensor::from_vec(pos, c, device).unwrap();
            last = Some(
                model
                    .forward_with_cache_last(&tok, &pos, &mut cache)
                    .unwrap(),
            );
            off += c;
        }
        bits(&last.unwrap())
    }

    fn run_embeds(
        model: &Gemma4,
        device: &Device,
        ids: &[u32],
        embeds: &Tensor,
        chunk: usize,
    ) -> Vec<u32> {
        let mut cache = model.new_kv_cache(64).unwrap();
        let mut last = None;
        let mut off = 0usize;
        while off < ids.len() {
            let c = chunk.min(ids.len() - off);
            let tok = Tensor::from_vec(ids[off..off + c].to_vec(), (1usize, c), device).unwrap();
            let pos: Vec<i32> = (off..off + c).map(|p| p as i32).collect();
            let pos = Tensor::from_vec(pos, c, device).unwrap();
            let e = embeds.narrow(0, off, c).unwrap();
            last = Some(
                model
                    .forward_with_cache_last_embeds(&tok, &e, &pos, &mut cache)
                    .unwrap(),
            );
            off += c;
        }
        bits(&last.unwrap())
    }

    #[test]
    fn dense_embeds_override_reproduces_the_token_path_bit_for_bit() {
        let device = cuda();
        let dir = temp_dir("dense");
        let model = tiny_dense(&dir.0, &device);

        let ids: Vec<u32> = vec![7, 33, 128, 4, 511, 90, 16, 2, 77];
        let embeds = embed_rows_as_mm_embeddings_does(
            model.embed_weight(),
            model.embed_scale() as f64,
            &ids,
            &device,
        );
        assert_eq!(embeds.dims(), [ids.len(), HIDDEN]);
        assert_eq!(embeds.dtype(), DType::F32);

        let want = run_tokens(&model, &device, &ids, ids.len());
        assert_eq!(
            want.len(),
            VOCAB,
            "forward_with_cache_last must keep returning one logit row"
        );
        assert_eq!(
            run_embeds(&model, &device, &ids, &embeds, ids.len()),
            want,
            "one-shot embeds prefill diverged from the token path"
        );
        assert_eq!(
            run_embeds(&model, &device, &ids, &embeds, 4),
            run_tokens(&model, &device, &ids, 4),
            "chunk-narrowed embeds prefill diverged from the same-chunking token path"
        );
    }

    #[test]
    fn dense_embeds_override_is_actually_read() {
        let device = cuda();
        let dir = temp_dir("dense_perturb");
        let model = tiny_dense(&dir.0, &device);
        let ids: Vec<u32> = vec![7, 33, 128, 4];
        let embeds = embed_rows_as_mm_embeddings_does(
            model.embed_weight(),
            model.embed_scale() as f64,
            &ids,
            &device,
        );
        let base = run_embeds(&model, &device, &ids, &embeds, ids.len());

        let mut rows: Vec<f32> = embeds.flatten_all().unwrap().to_vec1().unwrap();
        for x in rows.iter_mut().skip(HIDDEN).take(HIDDEN) {
            *x += 1.0;
        }
        let perturbed = Tensor::from_vec(rows, (ids.len(), HIDDEN), &device).unwrap();
        assert_ne!(
            run_embeds(&model, &device, &ids, &perturbed, ids.len()),
            base,
            "perturbing one embed row left the logits untouched: the override is being ignored"
        );

        let mut short = model.new_kv_cache(64).unwrap();
        let tok = Tensor::from_vec(ids.clone(), (1usize, ids.len()), &device).unwrap();
        let pos: Vec<i32> = (0..ids.len() as i32).collect();
        let pos = Tensor::from_vec(pos, ids.len(), &device).unwrap();
        let wrong = embeds.narrow(0, 0, ids.len() - 1).unwrap();
        assert!(
            model
                .forward_with_cache_last_embeds(&tok, &wrong, &pos, &mut short)
                .is_err(),
            "a [seq-1, hidden] override must be rejected, not silently broadcast"
        );
    }
}
