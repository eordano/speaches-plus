use anyhow::{bail, Context, Result};
use nv_weights::GgufLoader;

use crate::gemma4_moe::Gemma4MoeConfig;

pub fn gemma4_moe_config_from_gguf(g: &GgufLoader) -> Result<Gemma4MoeConfig> {
    let json = gemma4_moe_config_json_from_gguf(g)?;
    Gemma4MoeConfig::from_hf_json_str(&json)
        .context("build Gemma4MoeConfig from synthesized gguf config")
}

pub fn gemma4_moe_config_json_from_gguf(g: &GgufLoader) -> Result<String> {
    let arch = g.architecture().context("gguf general.architecture")?;
    if arch != "gemma4" {
        bail!("gguf architecture {arch:?} is not gemma4");
    }

    let block_count = g.md_u64("gemma4.block_count")? as usize;
    let hidden = g.md_u64("gemma4.embedding_length")? as usize;
    let ffn = g.md_u64("gemma4.feed_forward_length")? as usize;
    let n_head = g.md_u64("gemma4.attention.head_count")? as usize;

    let kv_list = match g.md_u64_list("gemma4.attention.head_count_kv") {
        Ok(v) => v,
        Err(list_err) => match g.md_u64("gemma4.attention.head_count_kv") {
            Ok(one) => vec![one],
            Err(_) => return Err(list_err),
        },
    };
    if kv_list.is_empty() {
        bail!("gguf head_count_kv list empty");
    }
    let kv_sliding = *kv_list.iter().max().unwrap() as usize;
    let kv_global = *kv_list.iter().min().unwrap() as usize;

    let head_dim = g.md_u64("gemma4.attention.key_length_swa")? as usize;
    let global_head_dim = g.md_u64("gemma4.attention.key_length")? as usize;

    let max_pos = g.md_u64("gemma4.context_length")? as usize;
    let rms_eps = g.md_f64("gemma4.attention.layer_norm_rms_epsilon")?;
    let sliding_window = g.md_u64("gemma4.attention.sliding_window")? as usize;

    let softcap = match g.md_f64("gemma4.final_logit_softcapping") {
        Ok(v) => v,
        Err(_) => match std::env::var("NV_GGUF_FINAL_LOGIT_SOFTCAP") {
            Ok(v) => v
                .parse::<f64>()
                .context("NV_GGUF_FINAL_LOGIT_SOFTCAP is not a float")?,
            Err(_) => bail!(
                "gguf has no `gemma4.final_logit_softcapping`; the HF gemma-4-26B-A4B config \
                 declares 30.0 and defaulting to 0.0 would silently disable logit softcapping. \
                 Set NV_GGUF_FINAL_LOGIT_SOFTCAP=<value> (30.0 to match HF, 0 to disable)."
            ),
        },
    };

    let rope_theta_global = g.md_f64("gemma4.rope.freq_base")?;
    let rope_theta_sliding = g.md_f64("gemma4.rope.freq_base_swa")?;

    let rope_dim_global = g
        .md_u64("gemma4.rope.dimension_count")
        .unwrap_or(global_head_dim as u64) as f64;
    let partial_rotary_factor = if global_head_dim > 0 {
        rope_dim_global / global_head_dim as f64
    } else {
        1.0
    };

    let pattern = g.md_bool_list("gemma4.attention.sliding_window_pattern")?;
    if pattern.len() != block_count {
        bail!(
            "gguf sliding_window_pattern len {} != block_count {}",
            pattern.len(),
            block_count
        );
    }
    let layer_types: Vec<&str> = pattern
        .iter()
        .map(|&sliding| {
            if sliding {
                "sliding_attention"
            } else {
                "full_attention"
            }
        })
        .collect();

    let num_experts = g.md_u64("gemma4.expert_count")? as usize;
    let top_k = g.md_u64("gemma4.expert_used_count")? as usize;
    let moe_inter = g.md_u64("gemma4.expert_feed_forward_length")? as usize;

    let vocab = g
        .gguf_tensor_shape("token_embd.weight")
        .and_then(|d| d.first().copied())
        .context("gguf token_embd.weight shape")?;

    let attention_k_eq_v = true;

    let tie_word_embeddings = !g.has_gguf_tensor("output.weight");

    let cfg_json = serde_json::json!({
        "model_type": "gemma4",
        "hidden_size": hidden,
        "intermediate_size": ffn,
        "num_hidden_layers": block_count,
        "num_attention_heads": n_head,
        "num_key_value_heads": kv_sliding,
        "num_global_key_value_heads": kv_global,
        "head_dim": head_dim,
        "global_head_dim": global_head_dim,
        "vocab_size": vocab,
        "max_position_embeddings": max_pos,
        "rms_norm_eps": rms_eps,
        "sliding_window": sliding_window,
        "final_logit_softcapping": softcap,
        "layer_types": layer_types,
        "attention_k_eq_v": attention_k_eq_v,
        "tie_word_embeddings": tie_word_embeddings,
        "hidden_activation": "gelu_pytorch_tanh",
        "rope_parameters": {
            "full_attention": {
                "rope_theta": rope_theta_global,
                "partial_rotary_factor": partial_rotary_factor,
            },
            "sliding_attention": {
                "rope_theta": rope_theta_sliding,
            },
        },
        "enable_moe_block": true,
        "num_experts": num_experts,
        "top_k_experts": top_k,
        "moe_intermediate_size": moe_inter,
    });

    Ok(cfg_json.to_string())
}
