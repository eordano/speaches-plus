use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device};
use nv_weights::gguf::{ggml_bytes, is_stacked_expert};
use nv_weights::{GgufLoader, TensorSource};
use std::path::PathBuf;

const ROOFLINE_GB_S: f64 = 738.5;

fn checkpoint() -> PathBuf {
    let raw = std::env::var("NV_GGUF_CKPT").unwrap_or_else(|_| {
        panic!(
            "NV_GGUF_CKPT is unset. Point it at a gemma4 MoE .gguf, e.g. the \
             gemma-4-26B-A4B file realized by the `darwin-llama` hub in \
             globals/hf-models.nix. This suite asserts against real weights and \
             has no synthetic fallback."
        )
    });
    let p = PathBuf::from(&raw);
    let md =
        std::fs::metadata(&p).unwrap_or_else(|e| panic!("NV_GGUF_CKPT={raw} is not readable: {e}"));
    assert!(
        md.len() > (1 << 30),
        "NV_GGUF_CKPT={raw} is {} bytes -- an LFS pointer stub, not a checkpoint. \
         The nix hub realizes only the files named in the pin's `filters.files`; \
         every other entry in the store directory is a 136-byte stub.",
        md.len()
    );
    p
}

fn open() -> GgufLoader {
    GgufLoader::open(checkpoint(), &Device::Cpu).expect("GgufLoader::open")
}

#[test]
#[ignore = "requires a real gemma4 MoE gguf via NV_GGUF_CKPT"]
fn gemma4_moe_gguf_header_and_quant_census() {
    let g = open();
    assert_eq!(g.architecture().expect("general.architecture"), "gemma4");

    let layers = g.md_u64("gemma4.block_count").expect("block_count") as usize;
    let hidden = g.md_u64("gemma4.embedding_length").expect("hidden") as usize;
    let experts = g.md_u64("gemma4.expert_count").expect("expert_count") as usize;
    let top_k = g.md_u64("gemma4.expert_used_count").expect("expert_used") as usize;
    let names = g.gguf_tensor_names();

    println!(
        "checkpoint {} :: {} layers, hidden {hidden}, {experts} experts top-{top_k}, {} tensors",
        g.general_name().unwrap_or_default(),
        layers,
        names.len()
    );

    assert!(layers > 0 && hidden > 0);
    assert!(top_k > 0 && top_k <= experts, "top_k {top_k} of {experts}");

    let census = g.quant_census();
    let total: usize = census.iter().map(|(_, _, b)| b).sum();
    for (dt, n, bytes) in &census {
        println!(
            "  {dt:?}: {n} tensors, {:.3} GB ({:.1}%)",
            *bytes as f64 / 1e9,
            100.0 * *bytes as f64 / total as f64
        );
    }
    assert_eq!(
        census.iter().map(|(_, n, _)| n).sum::<usize>(),
        names.len(),
        "census must cover every tensor"
    );

    for (dt, n, _) in &census {
        assert!(
            matches!(
                dt,
                GgmlDType::F32
                    | GgmlDType::F16
                    | GgmlDType::BF16
                    | GgmlDType::Q4_0
                    | GgmlDType::Q8_0
            ),
            "{n} tensors are {dt:?}; no kernel here decodes that layout"
        );
    }

    assert!(
        !g.has_gguf_tensor("output.weight"),
        "checkpoint ships an untied output.weight; the byte budget below assumes tied"
    );
    assert!(g.has_gguf_tensor("token_embd.weight"));
}

#[test]
#[ignore = "requires a real gemma4 MoE gguf via NV_GGUF_CKPT"]
fn gemma4_moe_gguf_name_map_covers_every_layer() {
    let g = open();
    let layers = g.md_u64("gemma4.block_count").unwrap() as usize;
    let mut missing = Vec::new();
    for i in 0..layers {
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "router.proj.weight",
            "experts.gate_up_proj",
            "experts.down_proj",
        ] {
            let name = format!("model.language_model.layers.{i}.{suffix}");
            if !g.has(&name) {
                missing.push(name);
            }
        }
    }
    assert!(missing.is_empty(), "unmapped gemma4 tensors: {missing:?}");
    for n in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ] {
        assert!(g.has(n), "missing {n}");
    }

    let sliding = g
        .md_bool_list("gemma4.attention.sliding_window_pattern")
        .expect("sliding_window_pattern");
    assert_eq!(sliding.len(), layers);
    for (i, is_sliding) in sliding.iter().enumerate() {
        let has_v = g.has(&format!(
            "model.language_model.layers.{i}.self_attn.v_proj.weight"
        ));
        assert_eq!(
            has_v, *is_sliding,
            "layer {i}: v_proj present={has_v} but sliding={is_sliding}"
        );
    }
}

#[test]
#[ignore = "requires a real gemma4 MoE gguf via NV_GGUF_CKPT"]
fn gemma4_moe_gguf_dequantizes_real_weights() {
    let g = open();
    let layers = g.md_u64("gemma4.block_count").unwrap() as usize;
    let hidden = g.md_u64("gemma4.embedding_length").unwrap() as usize;
    let experts = g.md_u64("gemma4.expert_count").unwrap() as usize;
    let last = layers - 1;

    for name in [
        "model.language_model.norm.weight".to_string(),
        format!("model.language_model.layers.{last}.self_attn.q_proj.weight"),
        format!("model.language_model.layers.{last}.mlp.down_proj.weight"),
        format!("model.language_model.layers.{last}.experts.down_proj"),
    ] {
        let t = g
            .get(&name, DType::F32)
            .unwrap_or_else(|e| panic!("dequantize {name}: {e:#}"));
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert!(!v.is_empty());
        assert!(
            v.iter().all(|x| x.is_finite()),
            "{name} produced non-finite values"
        );
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        assert!(
            var > 0.0,
            "{name} dequantized to a constant ({mean}) -- a zeroed or misread block"
        );
        println!("  {name} {:?} rms {:.5}", t.dims(), var.sqrt());
    }

    let exps = g
        .get(
            &format!("model.language_model.layers.{last}.experts.down_proj"),
            DType::F32,
        )
        .unwrap();
    assert_eq!(exps.rank(), 3, "stacked expert tensor must be rank-3");
    assert_eq!(exps.dims()[0], experts, "leading axis must be expert count");
    assert_eq!(exps.dims()[1], hidden, "down_proj out dim must be hidden");
}

#[test]
#[ignore = "requires a real gemma4 MoE gguf via NV_GGUF_CKPT"]
fn gemma4_moe_gguf_active_bytes_per_token_and_ceiling() {
    let g = open();
    let experts = g.md_u64("gemma4.expert_count").unwrap() as usize;
    let top_k = g.md_u64("gemma4.expert_used_count").unwrap() as usize;

    let active = g.active_bytes_per_token(top_k, experts).unwrap() as f64;
    let resident: usize = g.quant_census().iter().map(|(_, _, b)| b).sum();

    let mut check = 0f64;
    let mut by_group: std::collections::BTreeMap<&str, f64> = Default::default();
    for name in g.gguf_tensor_names() {
        let bytes = g.gguf_tensor_bytes(&name).unwrap() as f64;
        let (group, share) = if is_stacked_expert(&name) {
            ("experts", top_k as f64 / experts as f64)
        } else if name == "token_embd.weight" {
            ("lm_head (tied)", 1.0)
        } else if name.contains("attn_") && !name.contains("norm") {
            ("attn proj", 1.0)
        } else if name.contains("ffn_") && !name.contains("norm") {
            ("dense ffn + router", 1.0)
        } else {
            ("norms", 1.0)
        };
        check += bytes * share;
        *by_group.entry(group).or_default() += bytes * share;
    }
    assert!(
        (check - active).abs() / active < 1e-9,
        "active_bytes_per_token {active} disagrees with the tensor table {check}"
    );

    let ms = active / (ROOFLINE_GB_S * 1e9) * 1e3;
    println!(
        "resident checkpoint bytes : {:.3} GB",
        resident as f64 / 1e9
    );
    println!("ACTIVE bytes per token    : {:.3} GB", active / 1e9);
    for (grp, b) in &by_group {
        println!(
            "    {grp:<20} {:.3} GB ({:.1}%)",
            b / 1e9,
            100.0 * b / active
        );
    }
    println!(
        "floor @ {ROOFLINE_GB_S} GB/s        : {ms:.3} ms/tok = {:.1} tok/s",
        1e3 / ms
    );

    assert!(
        active < resident as f64,
        "MoE must read less than it stores"
    );
    assert!(active > 0.0);

    let mut served = 0f64;
    for name in g.gguf_tensor_names() {
        let dt = g.gguf_tensor_dtype(&name).unwrap();
        let elems = g.gguf_tensor_bytes(&name).unwrap() / dt.type_size() * dt.block_size();
        served += if is_stacked_expert(&name) && dt != GgmlDType::F32 {
            elems as f64 * 0.5625 * top_k as f64 / experts as f64
        } else if dt == GgmlDType::F32 {
            ggml_bytes(elems, GgmlDType::F32) as f64
        } else {
            elems as f64 * 2.0
        };
    }
    let ms_served = served / (ROOFLINE_GB_S * 1e9) * 1e3;
    println!(
        "wgpu bf16+w4-experts layout: {:.3} GB/tok -> {ms_served:.3} ms/tok = {:.1} tok/s ceiling",
        served / 1e9,
        1e3 / ms_served
    );
}

#[test]
#[ignore = "requires a real gemma4 MoE gguf via NV_GGUF_CKPT"]
fn gemma4_moe_gguf_carries_its_own_tokenizer_and_chat_template() {
    let g = open();

    let tokens = g
        .md_string_list("tokenizer.ggml.tokens")
        .expect("tokenizer.ggml.tokens");
    let merges = g
        .md_string_list("tokenizer.ggml.merges")
        .expect("tokenizer.ggml.merges");
    let template = g
        .md_string("tokenizer.chat_template")
        .expect("tokenizer.chat_template -- without it the engine refuses to serve");
    let eos = g
        .md_u64("tokenizer.ggml.eos_token_id")
        .expect("tokenizer.ggml.eos_token_id");
    let bos = g.md_u64("tokenizer.ggml.bos_token_id").expect("bos id");

    let vocab_from_tensor = g
        .gguf_tensor_shape("token_embd.weight")
        .and_then(|d| d.first().copied())
        .expect("token_embd shape");
    assert_eq!(
        tokens.len(),
        vocab_from_tensor,
        "the embedded tokenizer disagrees with the embedding matrix"
    );
    assert!(
        !merges.is_empty(),
        "a BPE tokenizer with no merges cannot be reconstructed"
    );
    assert!((eos as usize) < tokens.len() && (bos as usize) < tokens.len());
    assert!(
        template.contains("bos_token"),
        "the chat template does not reference bos_token, so tokenizer_config.json's \
         bos declaration would silently change the prompt head"
    );

    println!(
        "gguf tokenizer: {} tokens, {} merges, bos={bos} ({:?}) eos={eos} ({:?}), \
         chat_template {} bytes",
        tokens.len(),
        merges.len(),
        tokens[bos as usize],
        tokens[eos as usize],
        template.len()
    );

    let Ok(reference) = std::env::var("NV_GEMMA4_TOKENIZER_REF") else {
        panic!(
            "NV_GEMMA4_TOKENIZER_REF is unset. Point it at a shipped gemma-4 \
             tokenizer.json (e.g. the google/gemma-4-E4B-it snapshot). Without the \
             reference this test cannot tell a correct vocabulary from a plausible \
             one, and a serving dir assembled from the wrong tokenizer produces \
             fluent garbage rather than an error."
        )
    };
    let raw =
        std::fs::read_to_string(&reference).unwrap_or_else(|e| panic!("read {reference}: {e}"));
    let v: serde_json::Value = serde_json::from_str(&raw).expect("tokenizer.json parse");
    let model = &v["model"];
    let ref_vocab = model["vocab"].as_object().expect("model.vocab object");
    let ref_merges = model["merges"].as_array().expect("model.merges array");

    assert_eq!(ref_vocab.len(), tokens.len(), "vocab size differs");
    assert_eq!(ref_merges.len(), merges.len(), "merge count differs");

    let mut inv = vec![""; tokens.len()];
    for (tok, id) in ref_vocab {
        let id = id.as_u64().expect("vocab id") as usize;
        assert!(id < inv.len(), "reference id {id} out of range");
        inv[id] = tok.as_str();
    }
    let mismatched = (0..tokens.len()).filter(|&i| inv[i] != tokens[i]).count();
    assert_eq!(
        mismatched,
        0,
        "{mismatched} of {} token strings differ from {reference}",
        tokens.len()
    );

    let mut bad_merges = 0usize;
    for (i, m) in ref_merges.iter().enumerate() {
        let joined = match m {
            serde_json::Value::Array(p) => format!(
                "{} {}",
                p[0].as_str().expect("merge lhs"),
                p[1].as_str().expect("merge rhs")
            ),
            serde_json::Value::String(s) => s.clone(),
            other => panic!("unexpected merge encoding {other:?}"),
        };
        if joined != merges[i] {
            bad_merges += 1;
        }
    }
    assert_eq!(bad_merges, 0, "{bad_merges} merges differ from {reference}");

    println!(
        "gguf tokenizer is byte-identical to {reference} over {} tokens and {} merges",
        tokens.len(),
        merges.len()
    );
}
