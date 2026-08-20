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

struct Trace {
    toks: Vec<u32>,
    bits: Vec<u32>,
    v2: (usize, usize),
}

fn decode_trace(hidden: usize, inter: usize, vocab: usize, seed: u64) -> Trace {
    let config = Gemma4Config::from_hf_json_str(&config_json(hidden, inter, vocab)).unwrap();
    let weights = host_weights(&config, seed);
    let mut m = Gemma4Wgpu::new(config, &weights, 64).unwrap();
    let mut toks = Vec::new();
    let mut bits = Vec::new();
    for t in [3u32, 17, 5, 1, 11, 2, 9, 4] {
        let (tok, logits) = m.decode_step_logits(t % vocab as u32).unwrap();
        toks.push(tok);
        bits.extend(logits.iter().map(|v| v.to_bits()));
    }
    Trace {
        toks,
        bits,
        v2: m.nvfp4_v2_projections(),
    }
}

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn arm(env: &[(&str, &str)], hidden: usize, inter: usize, vocab: usize, seed: u64) -> Trace {
    let _g = env_lock();
    for (k, v) in env {
        std::env::set_var(k, v);
    }
    let out = decode_trace(hidden, inter, vocab, seed);
    for (k, _) in env {
        std::env::remove_var(k);
    }
    out
}

fn compare(name: &str, hidden: usize, inter: usize, vocab: usize, seed: u64) {
    let w8_off = ("NV_G4_WGPU_W8_FFN", "off");
    let base = arm(&[w8_off], hidden, inter, vocab, seed);
    let v2 = arm(
        &[w8_off, ("NV_WGPU_NVFP4_V2", "1")],
        hidden,
        inter,
        vocab,
        seed,
    );
    let tree = arm(
        &[w8_off, ("NV_WGPU_NVFP4_TREE", "1")],
        hidden,
        inter,
        vocab,
        seed,
    );
    let (old_toks, old_bits) = (base.toks, base.bits);
    let (new_toks, new_bits) = (v2.toks, v2.bits);
    let tree_toks = tree.toks;

    assert_eq!(
        base.v2.0, 0,
        "{name}: {} projections routed to v2 with NV_WGPU_NVFP4_V2 unset",
        base.v2.0
    );
    assert!(
        v2.v2.0 > 0,
        "{name}: NV_WGPU_NVFP4_V2=1 routed 0 of {} nvfp4 projections to v2, so this arm measures \
         the shipping path under a v2 label",
        v2.v2.1
    );
    assert_eq!(
        tree.v2.0, 0,
        "{name}: NV_WGPU_NVFP4_TREE=1 must exclude the v2 route"
    );

    let nonzero = old_bits.iter().filter(|w| **w != 0).count();
    assert!(
        nonzero * 4 >= old_bits.len() * 3,
        "{name}: degenerate trace, only {nonzero}/{} words nonzero",
        old_bits.len()
    );
    let logit_diff = old_bits
        .iter()
        .zip(new_bits.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "gemma4-nvfp4-v2 {name:<28} hidden={hidden} inter={inter} gate_up={}x{hidden} down={hidden}x{inter} logit-words={} differing={logit_diff}",
        2 * inter,
        old_bits.len()
    );
    assert_eq!(
        new_toks, old_toks,
        "{name}: v2 pk path and shipping sg path disagree on token ids"
    );
    assert_eq!(
        new_toks, tree_toks,
        "{name}: v2 pk path and tree path disagree on token ids"
    );
    assert_eq!(
        logit_diff,
        0,
        "{name}: v2 pk and shipping sg differ in {logit_diff} of {} logit words",
        old_bits.len()
    );
}

fn skip_unless_subgroups(what: &str) -> bool {
    let Some(ctx) = ctx_or_skip(what) else {
        return true;
    };
    if !nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2::subgroup32_ok(ctx) {
        eprintln!("skipping {what}: adapter subgroup width is not 32");
        return true;
    }
    let g = env_lock();
    assert!(
        !nv_models::gemma4_wgpu::nvfp4_v2_enabled(ctx),
        "{what}: NV_WGPU_NVFP4_V2 must be opt-in, not default-on"
    );
    std::env::set_var("NV_WGPU_NVFP4_V2", "1");
    let on = nv_models::gemma4_wgpu::nvfp4_v2_enabled(ctx);
    std::env::remove_var("NV_WGPU_NVFP4_V2");
    drop(g);
    if !on {
        eprintln!("skipping {what}: nvfp4 v2 path unavailable on this adapter");
        return true;
    }
    false
}

#[test]
fn boot_line_reports_engagement_not_just_the_knob() {
    use nv_models::gemma4_wgpu::nvfp4_v2_boot_line;
    assert_eq!(nvfp4_v2_boot_line(false, 0, 122), None);
    assert_eq!(nvfp4_v2_boot_line(false, 122, 122), None);
    assert!(nvfp4_v2_boot_line(true, 0, 122)
        .unwrap()
        .contains("requested but 0 of 122"));
    assert!(nvfp4_v2_boot_line(true, 122, 122)
        .unwrap()
        .contains("engaged on 122 of 122"));
}

#[test]
fn v2_mrow_and_fdec_routes_match_the_shipping_path_bit_for_bit() {
    if skip_unless_subgroups("v2_mrow_and_fdec_routes_match_the_shipping_path_bit_for_bit") {
        return;
    }
    for (n, k, want) in [(2048usize, 1024usize, "mrow"), (1024, 1024, "fdec")] {
        let route = nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2::select_pk(n, k)
            .map(|(_, _, e)| e)
            .unwrap_or("none");
        assert!(
            route.contains(want),
            "shape {n}x{k} routed to {route}, want {want}"
        );
    }
    compare("h1024_i1024_mrow_fdec", 1024, 1024, 128, 0x51a1);
}

#[test]
fn v2_warp_route_matches_on_shallow_and_ragged_row_counts() {
    if skip_unless_subgroups("v2_warp_route_matches_on_shallow_and_ragged_row_counts") {
        return;
    }
    assert_eq!(
        nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2::select_pk(544, 336)
            .map(|(_, _, e)| e)
            .unwrap_or("none"),
        "gemv_nvfp4_warp_pk"
    );
    compare("h336_i272_warp", 336, 272, 96, 0x51a2);
    compare("h144_i400_warp", 144, 400, 64, 0x51a3);
    compare("h272_i144_warp", 272, 144, 80, 0x51a4);
}
