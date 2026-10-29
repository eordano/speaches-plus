use candle_core::{Device, Tensor};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

pub const HIDDEN: usize = 128;
pub const INTER: usize = 64;
pub const LAYERS: usize = 3;
pub const HEAD_LAYERS: usize = 2;
pub const MAX_LEN: usize = 64;
const WORDS: [&str; 24] = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "choice", "noul", "score", "question", ":",
    "which", "team", "takes", "the", "item", "is", "red", "blue", "alpha", "beta", "big", "small",
    "old", "new",
];

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

pub fn write_tiny_checkpoint(dir: &Path) {
    std::fs::create_dir_all(dir.join("encoder")).unwrap();
    std::fs::create_dir_all(dir.join("tokenizer")).unwrap();
    let vocab: serde_json::Map<String, serde_json::Value> = WORDS
        .iter()
        .enumerate()
        .map(|(i, w)| (w.to_string(), json!(i)))
        .collect();
    let specials: Vec<serde_json::Value> = (0..5)
        .map(|i| json!({"id": i, "content": WORDS[i], "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}))
        .collect();
    let tokenizer = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": specials, "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    });
    std::fs::write(dir.join("tokenizer/tokenizer.json"), tokenizer.to_string()).unwrap();
    std::fs::write(
        dir.join("rl_agent_config.json"),
        json!({"max_len": MAX_LEN, "head_max_len": 32, "head_layers": HEAD_LAYERS}).to_string(),
    )
    .unwrap();
    let encoder = json!({"hidden_size": HIDDEN, "num_attention_heads": HIDDEN / 64, "num_hidden_layers": LAYERS, "local_attention": 8, "global_attn_every_n_layers": 3});
    std::fs::write(dir.join("encoder/config.json"), encoder.to_string()).unwrap();

    let mut rng = Lcg(7);
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut put = |name: String, shape: &[usize], gain: f32, offset: f32| {
        let data: Vec<f32> = (0..shape.iter().product())
            .map(|_| offset + gain * rng.next())
            .collect();
        tensors.insert(name, Tensor::from_vec(data, shape, &Device::Cpu).unwrap());
    };
    put(
        "encoder.embeddings.tok_embeddings.weight".into(),
        &[WORDS.len(), HIDDEN],
        1.0,
        0.0,
    );
    put("encoder.embeddings.norm.weight".into(), &[HIDDEN], 0.2, 1.0);
    put("encoder.final_norm.weight".into(), &[HIDDEN], 0.2, 1.0);
    for i in 0..LAYERS {
        let p = format!("encoder.layers.{i}");
        if i > 0 {
            put(format!("{p}.attn_norm.weight"), &[HIDDEN], 0.2, 1.0);
        }
        put(
            format!("{p}.attn.Wqkv.weight"),
            &[3 * HIDDEN, HIDDEN],
            0.6,
            0.0,
        );
        put(format!("{p}.attn.Wo.weight"), &[HIDDEN, HIDDEN], 0.3, 0.0);
        put(format!("{p}.mlp_norm.weight"), &[HIDDEN], 0.2, 1.0);
        put(format!("{p}.mlp.Wi.weight"), &[2 * INTER, HIDDEN], 0.3, 0.0);
        put(format!("{p}.mlp.Wo.weight"), &[HIDDEN, INTER], 0.3, 0.0);
    }
    for i in 0..HEAD_LAYERS {
        let p = format!("head.layers.{i}");
        for norm in ["norm1", "norm2"] {
            put(format!("{p}.{norm}.weight"), &[HIDDEN], 0.2, 1.0);
            put(format!("{p}.{norm}.bias"), &[HIDDEN], 0.2, 0.0);
        }
        put(
            format!("{p}.self_attn.in_proj_weight"),
            &[3 * HIDDEN, HIDDEN],
            0.6,
            0.0,
        );
        put(
            format!("{p}.self_attn.in_proj_bias"),
            &[3 * HIDDEN],
            0.2,
            0.0,
        );
        put(
            format!("{p}.self_attn.out_proj.weight"),
            &[HIDDEN, HIDDEN],
            0.3,
            0.0,
        );
        put(format!("{p}.self_attn.out_proj.bias"), &[HIDDEN], 0.2, 0.0);
        put(
            format!("{p}.linear1.weight"),
            &[4 * HIDDEN, HIDDEN],
            0.3,
            0.0,
        );
        put(format!("{p}.linear1.bias"), &[4 * HIDDEN], 0.2, 0.0);
        put(
            format!("{p}.linear2.weight"),
            &[HIDDEN, 4 * HIDDEN],
            0.3,
            0.0,
        );
        put(format!("{p}.linear2.bias"), &[HIDDEN], 0.2, 0.0);
    }
    put("type_emb.weight".into(), &[3, HIDDEN], 1.0, 0.0);
    put("scorer.0.weight".into(), &[HIDDEN], 0.2, 1.0);
    put("scorer.0.bias".into(), &[HIDDEN], 0.2, 0.0);
    put("scorer.1.weight".into(), &[HIDDEN, HIDDEN], 0.3, 0.0);
    put("scorer.1.bias".into(), &[HIDDEN], 0.2, 0.0);
    put("scorer.3.weight".into(), &[1, HIDDEN], 0.3, 0.0);
    put("scorer.3.bias".into(), &[1], 0.2, 0.0);
    put("act_head.0.weight".into(), &[16, HIDDEN + 4], 0.3, 0.0);
    put("act_head.0.bias".into(), &[16], 0.2, 0.0);
    put("act_head.2.weight".into(), &[2, 16], 0.3, 0.0);
    put("act_head.2.bias".into(), &[2], 0.2, 0.0);
    candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
}
