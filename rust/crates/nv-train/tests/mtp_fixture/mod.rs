use candle_core::{DType, Tensor};
use nv_train::qwen38::{mtp_expected_tensor_names, Q38Config, MTP_WEIGHTS_FILE_NAME};
use nv_train::qwen38_fixture::FixtureLcg;
use std::collections::HashMap;
use std::path::Path;

pub fn write_tiny_mtp_head(dir: &Path, cfg: &Q38Config) {
    let mut rng = FixtureLcg(0x717E_5EED);
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let q_rows = if cfg.attn_output_gate {
        cfg.num_attention_heads * hd * 2
    } else {
        cfg.num_attention_heads * hd
    };
    let kv_rows = cfg.num_key_value_heads * hd;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let mut put = |name: &str, x: Tensor| {
        t.insert(name.to_string(), x.to_dtype(DType::BF16).unwrap());
    };
    put("mtp.pre_fc_norm_embedding.weight", rng.tensor(&[h], 0.05));
    put("mtp.pre_fc_norm_hidden.weight", rng.tensor(&[h], 0.05));
    put("mtp.fc.weight", rng.tensor(&[h, 2 * h], 0.06));
    put("mtp.layers.0.input_layernorm.weight", rng.tensor(&[h], 0.05));
    put(
        "mtp.layers.0.post_attention_layernorm.weight",
        rng.tensor(&[h], 0.05),
    );
    put(
        "mtp.layers.0.self_attn.q_proj.weight",
        rng.tensor(&[q_rows, h], 0.06),
    );
    put(
        "mtp.layers.0.self_attn.k_proj.weight",
        rng.tensor(&[kv_rows, h], 0.06),
    );
    put(
        "mtp.layers.0.self_attn.v_proj.weight",
        rng.tensor(&[kv_rows, h], 0.06),
    );
    put(
        "mtp.layers.0.self_attn.o_proj.weight",
        rng.tensor(&[h, cfg.num_attention_heads * hd], 0.06),
    );
    put("mtp.layers.0.self_attn.q_norm.weight", rng.tensor(&[hd], 0.05));
    put("mtp.layers.0.self_attn.k_norm.weight", rng.tensor(&[hd], 0.05));
    put(
        "mtp.layers.0.mlp.gate_proj.weight",
        rng.tensor(&[cfg.intermediate_size, h], 0.06),
    );
    put(
        "mtp.layers.0.mlp.up_proj.weight",
        rng.tensor(&[cfg.intermediate_size, h], 0.06),
    );
    put(
        "mtp.layers.0.mlp.down_proj.weight",
        rng.tensor(&[h, cfg.intermediate_size], 0.06),
    );
    put("mtp.norm.weight", rng.tensor(&[h], 0.05));
    for name in mtp_expected_tensor_names() {
        assert!(
            t.contains_key(&name),
            "the fixture writer fell out of sync with mtp_expected_tensor_names: {name} missing"
        );
    }
    candle_core::safetensors::save(&t, dir.join(MTP_WEIGHTS_FILE_NAME)).unwrap();
}
