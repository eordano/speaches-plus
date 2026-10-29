use anyhow::{bail, Context, Result};
use candle_core::quantized::GgmlDType;
use nv_weights::GgufLoader;
use std::path::PathBuf;

use crate::gemma4_moe::Gemma4MoeConfig;

pub const G4MOE_SERVE_QAT_Q4_OPT_IN_ENV: &str = "NV_G4MOE_SERVE_QAT_Q4";
pub const QAT_Q4_GGUF_STAYS_OPT_IN_TOP1_ACC_0_4531_VS_BF16_0_5254_SAME_SESSION: &str =
    "the qat q4 gguf weights arm scored top-1 acc 0.4531 / ppl 32.344 against the same-session \
     bf16 arm's 0.5254 / 31.900 on g4moe_chat_wrapped_continuation_teacher_forced_ppl_via_wgpu_serving_decode \
     (private run ledger); accuracy \
     fell out of the recorded 0.5195-0.5254 band, so this weights source must never become the \
     silent default -- the suspected mechanism is double quantization, Q4_K dequant then the \
     wgpu w4 gs=32 requant";
pub const G4MOE_QAT_GGUF_ENV: &str = "NV_G4MOE_GGUF";
pub const G4MOE_QAT_GGUF_HUB_REPO: &str = "models--unsloth--gemma-4-26B-A4B-it-qat-GGUF";
pub const G4MOE_QAT_GGUF_FILE: &str = "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf";
pub const G4MOE_QAT_GGUF_HOME_CACHE_DIR_MATCHING_THE_GGUF_GEMMA26B_SUITES: &str =
    ".cache/gguf-gemma26b";
pub const G4MOE_GATE_UP_EXPERTS_CARRY_MOST_WEIGHT_BYTES_SO_THEY_PROVE_4BIT: &str =
    "blk.0.ffn_gate_up_exps.weight";

pub fn qat_q4_gguf_opted_in() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call(G4MOE_SERVE_QAT_Q4_OPT_IN_ENV)
}

pub fn qat_q4_weights_decision(
    opted_in: bool,
    verified_candidate: Option<PathBuf>,
) -> Option<PathBuf> {
    assert!(
        !QAT_Q4_GGUF_STAYS_OPT_IN_TOP1_ACC_0_4531_VS_BF16_0_5254_SAME_SESSION.is_empty(),
        "the opt-in rationale constant must state the failing numbers"
    );
    if !opted_in {
        return None;
    }
    verified_candidate
}

pub fn qat_q4_gguf_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var(G4MOE_QAT_GGUF_ENV) {
        out.push(PathBuf::from(p));
        return out;
    }
    if let Some(home) = std::env::var_os("HOME") {
        out.push(
            PathBuf::from(&home)
                .join(G4MOE_QAT_GGUF_HOME_CACHE_DIR_MATCHING_THE_GGUF_GEMMA26B_SUITES)
                .join(G4MOE_QAT_GGUF_FILE),
        );
    }
    let mut hubs: Vec<PathBuf> = Vec::new();
    if let Some(h) = std::env::var_os("HF_HUB_CACHE") {
        hubs.push(PathBuf::from(h));
    }
    if let Some(home) = std::env::var_os("HOME") {
        hubs.push(PathBuf::from(home).join(".cache/huggingface/hub"));
    }
    for hub in hubs {
        let snaps = hub.join(G4MOE_QAT_GGUF_HUB_REPO).join("snapshots");
        let Ok(rd) = std::fs::read_dir(&snaps) else {
            continue;
        };
        let mut cands: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path().join(G4MOE_QAT_GGUF_FILE))
            .collect();
        cands.sort();
        out.extend(cands);
    }
    out
}

pub fn verify_qat_q4_gguf_serves_this_snapshot(
    path: &std::path::Path,
    snapshot_cfg: &Gemma4MoeConfig,
) -> Result<()> {
    let g = GgufLoader::open(path, &candle_core::Device::Cpu)
        .with_context(|| format!("open qat gguf {}", path.display()))?;
    let dt = g
        .gguf_tensor_dtype(G4MOE_GATE_UP_EXPERTS_CARRY_MOST_WEIGHT_BYTES_SO_THEY_PROVE_4BIT)
        .with_context(|| {
            format!(
                "{} has no {G4MOE_GATE_UP_EXPERTS_CARRY_MOST_WEIGHT_BYTES_SO_THEY_PROVE_4BIT}",
                path.display()
            )
        })?;
    if !matches!(dt, GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q4K) {
        bail!(
            "{}: expert gate_up tensors are {dt:?}, not a 4-bit class; refusing to call this \
             checkpoint a 4-bit default",
            path.display()
        );
    }
    let gc = gemma4_moe_config_from_gguf(&g)?;
    let same = |name: &str, a: usize, b: usize| -> Result<()> {
        if a != b {
            bail!(
                "{}: gguf {name} {a} != served snapshot {name} {b}; this gguf is not the same \
                 model as the snapshot it would silently replace",
                path.display()
            );
        }
        Ok(())
    };
    same(
        "hidden_size",
        gc.base.hidden_size,
        snapshot_cfg.base.hidden_size,
    )?;
    same(
        "num_hidden_layers",
        gc.base.num_hidden_layers,
        snapshot_cfg.base.num_hidden_layers,
    )?;
    same(
        "vocab_size",
        gc.base.vocab_size,
        snapshot_cfg.base.vocab_size,
    )?;
    same("num_experts", gc.num_experts, snapshot_cfg.num_experts)?;
    same(
        "moe_intermediate_size",
        gc.moe_intermediate_size,
        snapshot_cfg.moe_intermediate_size,
    )?;
    Ok(())
}

pub fn opted_in_qat_q4_gguf_for_snapshot(snapshot_cfg: &Gemma4MoeConfig) -> Option<PathBuf> {
    if !qat_q4_gguf_opted_in() {
        return None;
    }
    let candidate = qat_q4_gguf_candidates().into_iter().find_map(|p| {
        if !p.is_file() {
            return None;
        }
        match verify_qat_q4_gguf_serves_this_snapshot(&p, snapshot_cfg) {
            Ok(()) => Some(p),
            Err(e) => {
                eprintln!(
                    "[g4moe-q4-optin] {} rejected: {e:#}; continuing with the given weights \
                     unless another candidate verifies",
                    p.display()
                );
                None
            }
        }
    });
    qat_q4_weights_decision(true, candidate)
}

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

#[cfg(test)]
mod qat_q4_default_tests {
    use super::*;

    #[test]
    fn the_qat_q4_decision_serves_the_gguf_only_when_opted_in_with_a_verified_candidate_and_the_env_override_short_circuits_discovery(
    ) {
        assert_eq!(
            qat_q4_weights_decision(false, Some(PathBuf::from("/x/q4.gguf"))),
            None
        );

        assert_eq!(qat_q4_weights_decision(true, None), None);

        assert_eq!(
            qat_q4_weights_decision(true, Some(PathBuf::from("/x/q4.gguf"))),
            Some(PathBuf::from("/x/q4.gguf"))
        );

        let prev = std::env::var_os(G4MOE_QAT_GGUF_ENV);
        unsafe { std::env::set_var(G4MOE_QAT_GGUF_ENV, "/override/only.gguf") };
        let cands = qat_q4_gguf_candidates();
        match prev {
            Some(v) => unsafe { std::env::set_var(G4MOE_QAT_GGUF_ENV, v) },
            None => unsafe { std::env::remove_var(G4MOE_QAT_GGUF_ENV) },
        }
        assert_eq!(cands, vec![PathBuf::from("/override/only.gguf")]);
    }
}
