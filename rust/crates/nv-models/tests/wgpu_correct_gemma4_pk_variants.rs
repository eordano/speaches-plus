#![cfg(feature = "wgpu")]

mod common;
use common::LcgCentered0p1Shift33 as Lcg;
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_wgpu::{
    quantize_nvfp4_host, Gemma4Wgpu, HostBf16Lin, HostLayer, HostProj, HostWeights,
};
use common::config_json_gemma4_hd64 as config_json;
use common::ctx_or_skip_quiet as ctx_or_skip;
use common::gemma4_host_weights_bf16_attn_nvfp4_ffn as host_weights;

fn decode_trace(hidden: usize, inter: usize, vocab: usize, seed: u64) -> (usize, Vec<u32>) {
    let config = Gemma4Config::from_hf_json_str(&config_json(hidden, inter, vocab)).unwrap();
    let weights = host_weights(&config, seed);
    let mut m = Gemma4Wgpu::new(config, &weights, 64).unwrap();
    let passes = m.pass_count();
    let mut bits = Vec::new();
    for t in [3u32, 17, 5, 1, 11, 2, 9, 4] {
        let (tok, logits) = m.decode_step_logits(t % vocab as u32).unwrap();
        bits.push(tok);
        bits.extend(logits.iter().map(|v| v.to_bits()));
    }
    (passes, bits)
}

fn compare_variants(name: &str, hidden: usize, inter: usize, vocab: usize, seed: u64) {
    std::env::set_var("NV_WGPU_NVFP4_TREE", "1");
    let (p_tree, tree) = decode_trace(hidden, inter, vocab, seed);
    std::env::remove_var("NV_WGPU_NVFP4_TREE");
    let (p_sg, sg) = decode_trace(hidden, inter, vocab, seed);

    assert_eq!(tree.len(), sg.len());
    let nonzero = tree.iter().filter(|w| **w != 0).count();
    assert!(
        nonzero * 4 >= tree.len() * 3,
        "{name}: degenerate trace, only {nonzero}/{} words nonzero",
        tree.len()
    );
    let diff = tree.iter().zip(sg.iter()).filter(|(a, b)| a != b).count();
    println!(
        "gemma4-pk variant-cmp {name:<26} hidden={hidden} inter={inter} vocab={vocab} passes tree={p_tree} sg={p_sg} words={} differing={diff}",
        tree.len()
    );
    assert_eq!(
        diff,
        0,
        "{name}: tree and sg nvfp4 decode paths differ in {diff} of {} words",
        tree.len()
    );
}

#[test]
fn pk_tree_and_sg_agree_when_rows_straddle_a_partial_128_scale_tile() {
    let Some(ctx) = ctx_or_skip("pk_tree_and_sg_agree_when_rows_straddle_a_partial_128_scale_tile")
    else {
        return;
    };
    if !nv_kernels::wgpu_backend::kernels::gemv_nvfp4::subgroup_ok(ctx) {
        eprintln!("skipping: adapter has no fixed-32 subgroups, tree == sg trivially");
        return;
    }
    compare_variants("hidden336_inter272", 336, 272, 96, 0xabc1);
    compare_variants("hidden144_inter400", 144, 400, 64, 0xabc2);
    compare_variants("hidden512_inter128", 512, 128, 64, 0xabc3);
}

#[test]
fn pk_tree_and_sg_agree_when_rows_are_not_a_multiple_of_the_subgroup_row_block() {
    let Some(ctx) =
        ctx_or_skip("pk_tree_and_sg_agree_when_rows_are_not_a_multiple_of_the_subgroup_row_block")
    else {
        return;
    };
    if !nv_kernels::wgpu_backend::kernels::gemv_nvfp4::subgroup_ok(ctx) {
        eprintln!("skipping: adapter has no fixed-32 subgroups, tree == sg trivially");
        return;
    }
    for (hidden, inter) in [(272usize, 144usize), (400, 208), (176, 336)] {
        assert_eq!(hidden % 16, 0);
        assert_eq!(inter % 16, 0);
        compare_variants(
            &format!("h{hidden}_i{inter}_rows_mod4={}", hidden % 4),
            hidden,
            inter,
            80,
            0xdef0 + hidden as u64,
        );
    }
}
