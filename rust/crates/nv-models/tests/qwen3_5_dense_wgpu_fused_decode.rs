#![cfg(feature = "wgpu")]

mod common;
use common::have_gpu;
use common::tiny_config_q3d_mixed_layers as tiny_config;
use common::tiny_weights_q3d as tiny_weights;
use common::EnvPins;
use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_moe::LayerType;

const STEPS: [u32; 6] = [3, 11, 5, 40, 2, 19];

const DN_FUSION_REMOVES_SPLIT_GATING_RECURRENT_OUT_MINUS_THE_FUSED_ONE: usize = 3;

const ATTN_FUSION_REMOVES_KNORM_AND_QCAST: usize = 2;

const KVW_FUSION_REMOVES_KV_WRITE_AND_ONE_FP8_QUANTIZER_PER_ATTN_LAYER: usize = 2;

fn stream(envs: &[(&'static str, Option<&str>)]) -> (usize, Vec<(u32, Vec<u32>)>) {
    let _pins = EnvPins::pin(envs);
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xfeed_5eed_0001);
    let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg, &hw, 32).expect("build wgpu model");
    let mut out = Vec::new();
    for &t in &STEPS {
        let (arg, logits) = gpu.decode_step_logits(t).expect("decode step");
        out.push((arg, logits.iter().map(|x| x.to_bits()).collect()));
    }
    (gpu.pass_count(), out)
}

fn assert_streams_bit_identical(arm: &str, base: &[(u32, Vec<u32>)], got: &[(u32, Vec<u32>)]) {
    assert_eq!(base.len(), got.len(), "{arm}: step count");
    for (i, ((ba, bl), (ga, gl))) in base.iter().zip(got.iter()).enumerate() {
        assert_eq!(ba, ga, "{arm}: argmax diverged at step {i}");
        assert_eq!(bl.len(), gl.len(), "{arm}: logit width at step {i}");
        for (j, (b, g)) in bl.iter().zip(gl.iter()).enumerate() {
            assert_eq!(
                b,
                g,
                "{arm}: logit bits diverged at step {i} index {j} ({} vs {}); the fused decode \
                 arms preserve per-element arithmetic order, so anything short of bit identity \
                 is a fusion defect, not rounding",
                f32::from_bits(*b),
                f32::from_bits(*g)
            );
        }
    }
}

#[test]
fn fused_decode_envs_are_bit_identical_to_the_default_graph_and_remove_the_promised_dispatches() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let dn_layers = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::LinearAttention))
        .count();
    let attn_layers = cfg.num_hidden_layers - dn_layers;
    assert!(
        dn_layers >= 2 && attn_layers >= 1,
        "the tiny config no longer mixes DN and full-attention layers, so this identity would \
         not exercise both fused arms"
    );

    let off: [(&'static str, Option<&str>); 6] = [
        ("NV_Q3D_FUSE_DN", Some("0")),
        ("NV_Q3D_FUSE_ATTN", Some("0")),
        ("NV_Q3D_FUSE_DN_GEMV", Some("0")),
        ("NV_Q3D_FUSE_MLP", Some("0")),
        ("NV_Q3D_FUSE_MLP_GEMV", Some("0")),
        ("NV_Q3D_FUSE_KVW", Some("0")),
    ];
    let (base_passes, base) = stream(&off);

    let (dn_passes, dn) = stream(&[
        ("NV_Q3D_FUSE_DN", Some("1")),
        ("NV_Q3D_FUSE_ATTN", Some("0")),
        ("NV_Q3D_FUSE_DN_GEMV", Some("0")),
        ("NV_Q3D_FUSE_MLP", Some("0")),
        ("NV_Q3D_FUSE_MLP_GEMV", Some("0")),
        ("NV_Q3D_FUSE_KVW", Some("0")),
    ]);
    assert_streams_bit_identical("NV_Q3D_FUSE_DN", &base, &dn);
    assert_eq!(
        base_passes - dn_passes,
        dn_layers * DN_FUSION_REMOVES_SPLIT_GATING_RECURRENT_OUT_MINUS_THE_FUSED_ONE,
        "NV_Q3D_FUSE_DN must remove split+gating+recurrent+out minus the one fused dispatch on \
         every DN layer ({dn_layers} DN layers, base {base_passes} passes, fused {dn_passes})"
    );

    let (attn_passes, attn) = stream(&[
        ("NV_Q3D_FUSE_DN", Some("0")),
        ("NV_Q3D_FUSE_ATTN", Some("1")),
        ("NV_Q3D_FUSE_DN_GEMV", Some("0")),
        ("NV_Q3D_FUSE_MLP", Some("0")),
        ("NV_Q3D_FUSE_MLP_GEMV", Some("0")),
        ("NV_Q3D_FUSE_KVW", Some("0")),
    ]);
    assert_streams_bit_identical("NV_Q3D_FUSE_ATTN", &base, &attn);
    assert_eq!(
        base_passes - attn_passes,
        attn_layers * ATTN_FUSION_REMOVES_KNORM_AND_QCAST,
        "NV_Q3D_FUSE_ATTN must remove knorm and qcast on every full-attention layer \
         ({attn_layers} attn layers, base {base_passes} passes, fused {attn_passes})"
    );

    let (gemv_passes, gemv) = stream(&[
        ("NV_Q3D_FUSE_DN", Some("0")),
        ("NV_Q3D_FUSE_ATTN", Some("0")),
        ("NV_Q3D_FUSE_DN_GEMV", Some("1")),
        ("NV_Q3D_FUSE_MLP", Some("0")),
        ("NV_Q3D_FUSE_MLP_GEMV", Some("0")),
        ("NV_Q3D_FUSE_KVW", Some("0")),
    ]);
    assert_streams_bit_identical("NV_Q3D_FUSE_DN_GEMV", &base, &gemv);
    assert_eq!(
        base_passes, gemv_passes,
        "tiny host weights carry no fp8 DN projections, so the merged DN gemv route must not \
         engage here; its bit gate on real fp8 shapes is \
         q3w_gemv_dn_merged_is_bit_identical_to_the_three_projection_dispatches in \
         graph_q3d_fused_decode_identity"
    );

    let (mlp_passes, mlp) = stream(&[
        ("NV_Q3D_FUSE_DN", Some("0")),
        ("NV_Q3D_FUSE_ATTN", Some("0")),
        ("NV_Q3D_FUSE_DN_GEMV", Some("0")),
        ("NV_Q3D_FUSE_MLP", Some("1")),
        ("NV_Q3D_FUSE_MLP_GEMV", Some("0")),
        ("NV_Q3D_FUSE_KVW", Some("0")),
    ]);
    assert_streams_bit_identical("NV_Q3D_FUSE_MLP", &base, &mlp);
    assert_eq!(
        base_passes, mlp_passes,
        "tiny host weights carry a bf16 MLP, so the fused silu+down-quant route (nvfp4 down \
         only) must not engage here; its bit gate on nvfp4 shapes is \
         q3w_silu_mul_quant_is_bit_identical_to_silu_mul_then_quant_rows in \
         graph_q3d_fused_decode_identity"
    );

    let (kvw_passes, kvw) = stream(&[
        ("NV_Q3D_FUSE_DN", Some("0")),
        ("NV_Q3D_FUSE_ATTN", Some("0")),
        ("NV_Q3D_FUSE_DN_GEMV", Some("0")),
        ("NV_Q3D_FUSE_MLP", Some("0")),
        ("NV_Q3D_FUSE_MLP_GEMV", Some("0")),
        ("NV_Q3D_FUSE_KVW", Some("1")),
    ]);
    assert_streams_bit_identical("NV_Q3D_FUSE_KVW", &base, &kvw);
    assert_eq!(
        base_passes - kvw_passes,
        attn_layers * KVW_FUSION_REMOVES_KV_WRITE_AND_ONE_FP8_QUANTIZER_PER_ATTN_LAYER,
        "NV_Q3D_FUSE_KVW must fold kv_write and the fp8 k/v quantize pair into one dispatch on \
         every full-attention layer ({attn_layers} attn layers, base {base_passes} passes, \
         fused {kvw_passes})"
    );

    let (mlp_gemv_passes, mlp_gemv) = stream(&[
        ("NV_Q3D_FUSE_DN", Some("0")),
        ("NV_Q3D_FUSE_ATTN", Some("0")),
        ("NV_Q3D_FUSE_DN_GEMV", Some("0")),
        ("NV_Q3D_FUSE_MLP", Some("0")),
        ("NV_Q3D_FUSE_MLP_GEMV", Some("1")),
        ("NV_Q3D_FUSE_KVW", Some("0")),
    ]);
    assert_streams_bit_identical("NV_Q3D_FUSE_MLP_GEMV", &base, &mlp_gemv);
    assert_eq!(
        base_passes, mlp_gemv_passes,
        "tiny host weights carry a bf16 MLP, so the merged nvfp4 gate+up gemv (mrow2 route \
         only) must not engage here; its bit gate on nvfp4 shapes is \
         q3w_gemv_nvfp4_mrow2_2w_is_bit_identical_to_the_gate_and_up_mrow2_dispatches in \
         graph_q3d_fused_decode_identity"
    );

    let (all_passes, all) = stream(&[
        ("NV_Q3D_FUSE_DN", Some("1")),
        ("NV_Q3D_FUSE_ATTN", Some("1")),
        ("NV_Q3D_FUSE_DN_GEMV", Some("1")),
        ("NV_Q3D_FUSE_MLP", Some("1")),
        ("NV_Q3D_FUSE_MLP_GEMV", Some("1")),
        ("NV_Q3D_FUSE_KVW", Some("1")),
    ]);
    assert_streams_bit_identical("all fusion envs", &base, &all);
    assert_eq!(
        base_passes - all_passes,
        dn_layers * DN_FUSION_REMOVES_SPLIT_GATING_RECURRENT_OUT_MINUS_THE_FUSED_ONE
            + attn_layers * ATTN_FUSION_REMOVES_KNORM_AND_QCAST
            + attn_layers * KVW_FUSION_REMOVES_KV_WRITE_AND_ONE_FP8_QUANTIZER_PER_ATTN_LAYER,
        "the fused arms must compose (base {base_passes}, all-fused {all_passes})"
    );
    eprintln!(
        "[q3d-fused-decode] bit-identical across arms; passes/token base={base_passes} \
         dn={dn_passes} attn={attn_passes} gemv={gemv_passes} mlp={mlp_passes} \
         kvw={kvw_passes} mlp_gemv={mlp_gemv_passes} all={all_passes}"
    );
}
