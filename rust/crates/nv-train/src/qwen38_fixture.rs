use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::qwen38::{layer_prefix, LayerKind, Q38Config, EMBED_TOKENS, FINAL_NORM, LM_HEAD};

pub struct FixtureLcg(pub u64);

impl FixtureLcg {
    pub fn next_f32(&mut self, std: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        u * std * 1.732
    }

    pub fn tensor(&mut self, shape: &[usize], std: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| self.next_f32(std)).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }
}

pub fn tiny_config_json() -> String {
    tiny_config_json_with_head_dim(8)
}

pub fn tiny_config_json_with_head_dim(head_dim: u32) -> String {
    serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": head_dim,
            "intermediate_size": 64,
            "vocab_size": 128,
            "rms_norm_eps": 1e-6,
            "partial_rotary_factor": 0.5,
            "attn_output_gate": true,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention"
            ],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": { "rope_theta": 10000.0, "partial_rotary_factor": 0.5 }
        }
    })
    .to_string()
}

pub fn write_tiny_checkpoint(dir: &Path) -> Q38Config {
    write_tiny_checkpoint_with_head_dim(dir, 8)
}

pub fn write_tiny_checkpoint_with_head_dim(dir: &Path, head_dim: u32) -> Q38Config {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let cfg_json = tiny_config_json_with_head_dim(head_dim);
    std::fs::write(dir.join("config.json"), &cfg_json).unwrap();
    let cfg = Q38Config::from_hf_json_str(&cfg_json).unwrap();

    let mut rng = FixtureLcg(0x5EED_1234);
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let h = cfg.hidden_size;
    let mut put = |name: String, x: Tensor| {
        t.insert(name, x.to_dtype(DType::BF16).unwrap());
    };
    put(
        EMBED_TOKENS.to_string(),
        rng.tensor(&[cfg.vocab_size, h], 0.05),
    );
    put(FINAL_NORM.to_string(), rng.tensor(&[h], 0.05));
    put(
        format!("{LM_HEAD}.weight"),
        rng.tensor(&[cfg.vocab_size, h], 0.05),
    );
    for li in 0..cfg.num_hidden_layers {
        let p = layer_prefix(li);
        put(
            format!("{p}.input_layernorm.weight"),
            rng.tensor(&[h], 0.05),
        );
        put(
            format!("{p}.post_attention_layernorm.weight"),
            rng.tensor(&[h], 0.05),
        );
        put(
            format!("{p}.mlp.gate_proj.weight"),
            rng.tensor(&[cfg.intermediate_size, h], 0.06),
        );
        put(
            format!("{p}.mlp.up_proj.weight"),
            rng.tensor(&[cfg.intermediate_size, h], 0.06),
        );
        put(
            format!("{p}.mlp.down_proj.weight"),
            rng.tensor(&[h, cfg.intermediate_size], 0.06),
        );
        match cfg.layer_types[li] {
            LayerKind::FullAttention => {
                let hd = cfg.head_dim;
                put(
                    format!("{p}.self_attn.q_proj.weight"),
                    rng.tensor(&[cfg.num_attention_heads * hd * 2, h], 0.06),
                );
                put(
                    format!("{p}.self_attn.k_proj.weight"),
                    rng.tensor(&[cfg.num_key_value_heads * hd, h], 0.06),
                );
                put(
                    format!("{p}.self_attn.v_proj.weight"),
                    rng.tensor(&[cfg.num_key_value_heads * hd, h], 0.06),
                );
                put(
                    format!("{p}.self_attn.o_proj.weight"),
                    rng.tensor(&[h, cfg.num_attention_heads * hd], 0.06),
                );
                put(
                    format!("{p}.self_attn.q_norm.weight"),
                    rng.tensor(&[hd], 0.05),
                );
                put(
                    format!("{p}.self_attn.k_norm.weight"),
                    rng.tensor(&[hd], 0.05),
                );
            }
            LayerKind::LinearAttention => {
                let lac = cfg.linear_attn_config();
                let n_v = lac.linear_num_value_heads;
                put(
                    format!("{p}.linear_attn.in_proj_qkv.weight"),
                    rng.tensor(&[lac.conv_dim(), h], 0.06),
                );
                put(
                    format!("{p}.linear_attn.in_proj_z.weight"),
                    rng.tensor(&[lac.value_dim(), h], 0.06),
                );
                put(
                    format!("{p}.linear_attn.in_proj_a.weight"),
                    rng.tensor(&[n_v, h], 0.06),
                );
                put(
                    format!("{p}.linear_attn.in_proj_b.weight"),
                    rng.tensor(&[n_v, h], 0.06),
                );
                put(
                    format!("{p}.linear_attn.conv1d.weight"),
                    rng.tensor(&[lac.conv_dim(), 1, lac.linear_conv_kernel_dim], 0.3),
                );
                put(format!("{p}.linear_attn.A_log"), rng.tensor(&[n_v], 0.1));
                put(format!("{p}.linear_attn.dt_bias"), rng.tensor(&[n_v], 0.1));
                put(
                    format!("{p}.linear_attn.norm.weight"),
                    rng.tensor(&[lac.linear_value_head_dim], 0.02)
                        .affine(1.0, 1.0)
                        .unwrap(),
                );
                put(
                    format!("{p}.linear_attn.out_proj.weight"),
                    rng.tensor(&[h, lac.value_dim()], 0.06),
                );
            }
        }
    }
    candle_core::safetensors::save(&t, dir.join("model.safetensors")).unwrap();
    cfg
}
