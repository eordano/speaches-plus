#![cfg(feature = "laguna-wip")]

use nv_models::laguna::LagunaConfig;
use nv_models::laguna_wgpu::{window_start, FfnKind, GateKind, LagunaShapes, LayerType};

const TINY_CONFIG: &str = r#"{
    "architectures": ["LagunaForCausalLM"],
    "model_type": "laguna",
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 16,
    "norm_topk_prob": true,
    "mlp_only_layers": [0],
    "decoder_sparse_step": 1,
    "tie_word_embeddings": false,
    "gating": "per-head",
    "sliding_window": 8,
    "moe_routed_scaling_factor": 2.5,
    "eos_token_id": [2, 24],
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
    "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "num_attention_heads_per_layer": [4, 4, 8, 4]
}"#;

fn shapes(max_seq: usize) -> LagunaShapes {
    let cfg = LagunaConfig::from_hf_json_str(TINY_CONFIG).unwrap();
    LagunaShapes::derive(&cfg, max_seq).unwrap()
}

#[test]
fn derives_hybrid_attention_and_ffn_kinds() {
    let s = shapes(64);
    assert_eq!(s.num_layers, 4);
    assert_eq!(s.layer(0).attn_kind, LayerType::FullAttention);
    assert_eq!(s.layer(1).attn_kind, LayerType::SlidingAttention);
    assert_eq!(s.layer(0).ffn_kind, FfnKind::Dense);
    assert_eq!(s.layer(1).ffn_kind, FfnKind::Moe);
    assert_eq!(s.moe_layer_indices(), vec![1, 2, 3]);
    assert_eq!(s.dense_layer_indices(), vec![0]);
    assert_eq!(s.layer(0).ffn_intermediate, 64);
    assert_eq!(s.layer(1).ffn_intermediate, 16);
}

#[test]
fn derives_non_uniform_head_counts() {
    let s = shapes(64);
    assert_eq!(s.layer(0).num_q_heads, 4);
    assert_eq!(s.layer(2).num_q_heads, 8);
    assert_eq!(s.max_q_heads, 8);
    assert_eq!(s.layer(0).gqa_group, 2);
    assert_eq!(s.layer(2).gqa_group, 4);
    assert_eq!(s.layer(0).q_rows, 32);
    assert_eq!(s.layer(2).q_rows, 64);
    assert_eq!(s.layer(0).kv_rows, 16);
    assert_eq!(s.layer(2).kv_rows, 16);
}

#[test]
fn per_head_gating_rows_track_layer_head_count() {
    let s = shapes(64);
    assert_eq!(s.gate_kind, GateKind::PerHead);
    assert_eq!(s.layer(0).gate_rows, 4);
    assert_eq!(s.layer(2).gate_rows, 8);
    assert_eq!(GateKind::None.rows_for(8, 8), 0);
    assert_eq!(GateKind::PerElement.rows_for(8, 8), 64);
}

#[test]
fn partial_rope_geometry_is_per_attention_flavour() {
    let s = shapes(64);
    assert_eq!(s.rotary_dim_full, 4);
    assert_eq!(s.rotary_dim_sliding, 8);
    assert_eq!(s.layer(0).rotary_dim, 4);
    assert_eq!(s.layer(1).rotary_dim, 8);
    assert_eq!(s.rope_inv_freq_full.len(), 2);
    assert_eq!(s.rope_inv_freq_sliding.len(), 4);
    assert!((s.layer(0).rope_out_scale - 1.3465736).abs() < 1e-6);
    assert!((s.layer(1).rope_out_scale - 1.0).abs() < 1e-6);
}

#[test]
fn sliding_layers_carry_a_window_and_full_layers_do_not() {
    let s = shapes(64);
    assert_eq!(s.layer(0).window_tokens, None);
    assert_eq!(s.layer(1).window_tokens, Some(8));
    assert_eq!(window_start(4, Some(8)), 0);
    assert_eq!(window_start(8, Some(8)), 0);
    assert_eq!(window_start(9, Some(8)), 1);
    assert_eq!(window_start(9, None), 0);
}

#[test]
fn kv_geometry_is_uniform_across_layers() {
    let s = shapes(64);
    for l in &s.layers {
        assert_eq!(l.kv_capacity_tokens, 64);
        assert_eq!(l.num_kv_heads, 2);
        assert_eq!(l.head_dim, 8);
    }
    assert_eq!(s.kv_cache_bytes_bf16(), 4 * 2 * (64 * 16 * 2));
}

#[test]
fn softmax_scale_uses_head_dim() {
    let s = shapes(64);
    let want = (8f32).powf(-0.5);
    for l in &s.layers {
        assert!((l.attn_softmax_scale - want).abs() < 1e-7);
    }
}

#[test]
fn zero_max_seq_is_rejected() {
    let cfg = LagunaConfig::from_hf_json_str(TINY_CONFIG).unwrap();
    assert!(LagunaShapes::derive(&cfg, 0).is_err());
}

#[test]
fn nvfp4_alignment_gate_rejects_unaligned_tiny_config() {
    let s = shapes(64);
    assert!(s.validate_nvfp4_alignment().is_err());
}
