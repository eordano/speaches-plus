#![cfg(feature = "laguna-wip")]

mod common;
use common::bit_diff;
use common::CFG;
use common::have_gpu;
mod hub_snapshot;

use nv_models::laguna::LagunaConfig;
use nv_models::laguna_wgpu::config::LagunaShapes;
use nv_models::laguna_wgpu::weights::random_host_weights;
use nv_models::laguna_wgpu::{HostWeights, LagunaWgpu};

const MAX_SEQ: usize = 64;

const CHAIN_WIDTHS_COVER_ONE_ROW_A_SHORT_CHAIN_AND_A_WIDE_CHAIN: [usize; 3] = [1, 3, 8];

const LAGUNA_VERIFY_IS_SUBMISSION_BATCHING_NOT_AN_M_ROW_MATMUL: &str =
    "LagunaWgpu::verify_chain encodes k per-token pass lists plus k argmax copies into ONE queue \
     submission, exactly as prefill_chunk_one_submission_of_per_token_passes does for the trunk. \
     Every row therefore runs the decode kernels with the decode uniforms, so bit-identity with \
     one-at-a-time stepping is structural rather than measured; what this suite proves is that \
     the uniform list, the KV positions and the per-row token copies line up. The cache is \
     linear (kv_capacity_tokens == max_seq_tokens) and every row bounds its own read by its own \
     total, so rows past the accepted prefix are never read and advance(n) is a pure commit.";

fn config() -> LagunaConfig {
    LagunaConfig::from_hf_json_str(CFG).unwrap()
}

fn weights(cfg: &LagunaConfig, seed: u64) -> HostWeights {
    let shapes = LagunaShapes::derive(cfg, MAX_SEQ).unwrap();
    random_host_weights(&shapes, seed)
}

fn ids(cfg: &LagunaConfig, n: usize, salt: u32) -> Vec<u32> {
    (0..n)
        .map(|i| ((i as u32 * 7 + salt * 13 + 1) % (cfg.vocab_size as u32 - 1)) + 1)
        .collect()
}

fn primed(cfg: &LagunaConfig, hw: &HostWeights, prompt: &[u32]) -> LagunaWgpu {
    let mut m = LagunaWgpu::new(cfg.clone(), hw, MAX_SEQ).expect("build model");
    for t in prompt {
        m.prefill_step(*t).expect("prime with per-token prefill");
    }
    m
}

#[test]
fn verify_chain_rows_are_the_same_tokens_stepped_one_at_a_time() {
    assert!(
        LAGUNA_VERIFY_IS_SUBMISSION_BATCHING_NOT_AN_M_ROW_MATMUL.contains("ONE queue submission"),
        "the design note this suite is written against went missing"
    );
    if !have_gpu() {
        panic!("needs a wgpu adapter; a skipped identity proof reads as a passed one");
    }
    let cfg = config();
    let hw = weights(&cfg, 0xC0FFEE);
    let prompt = ids(&cfg, 11, 1);
    let rows = primed(&cfg, &hw, &prompt).verify_max_rows();
    assert!(
        rows >= 2,
        "LagunaWgpu::verify_max_rows() is {rows}: without a multi-token verify entry every \
         speculative round degrades to one token per submission"
    );

    let mut seen: Vec<(usize, Vec<u32>)> = Vec::new();
    for k in CHAIN_WIDTHS_COVER_ONE_ROW_A_SHORT_CHAIN_AND_A_WIDE_CHAIN {
        let k = k.min(rows);
        let chain = ids(&cfg, k, 5);
        let mut chained = primed(&cfg, &hw, &prompt);
        let pos0 = chained.current_pos();
        let got = chained.verify_chain(&chain).expect("verify_chain");
        assert_eq!(
            chained.current_pos(),
            pos0,
            "verify_chain committed {k} rows by itself; commit belongs to advance(n) so that a \
             partial accept costs nothing"
        );

        let mut stepped = primed(&cfg, &hw, &prompt);
        let want: Vec<u32> = chain
            .iter()
            .map(|t| stepped.decode_step(*t).expect("decode step"))
            .collect();
        assert_eq!(
            got, want,
            "k={k}: the batched verify submission and one-at-a-time stepping disagree. Both run \
             the same pass list with the same uniforms, so a difference is the uniform record \
             layout, the per-row token copy offset, or a KV position off by one"
        );
        seen.push((k, got));
    }
    let widest = seen
        .iter()
        .max_by_key(|(k, _)| *k)
        .expect("at least one chain width ran");
    assert!(
        widest.1.iter().any(|t| *t != widest.1[0]),
        "the {}-row chain emitted the same token on every row {:?}; a verify entry that returns \
         the last row's argmax for every row would pass the equality assertion on a model whose \
         logits barely move, so this oracle refuses that shape",
        widest.0,
        widest.1
    );
    eprintln!("[laguna-verify] (k, argmax rows) = {seen:?}");
}

#[test]
fn accepting_a_prefix_of_a_verified_chain_leaves_the_pure_one_token_stream() {
    if !have_gpu() {
        panic!("needs a wgpu adapter; a skipped losslessness proof reads as a passed one");
    }
    let cfg = config();
    let hw = weights(&cfg, 0x5EED);
    let prompt = ids(&cfg, 9, 2);
    let mut spec = primed(&cfg, &hw, &prompt);
    let rows = spec.verify_max_rows();
    let k = rows.min(6);
    assert!(
        k >= 3,
        "verify width {rows} is too narrow for a partial accept"
    );
    let accept = k / 2;
    let chain = ids(&cfg, k, 4);
    spec.verify_chain(&chain).expect("verify_chain");
    spec.advance(accept).expect("advance the accepted prefix");
    assert_eq!(
        spec.current_pos(),
        prompt.len() + accept,
        "advance(n) must move exactly n rows"
    );

    let mut plain = primed(&cfg, &hw, &prompt);
    for t in &chain[..accept] {
        plain.decode_step(*t).expect("step the accepted prefix");
    }

    for (i, t) in ids(&cfg, 5, 7).into_iter().enumerate() {
        let (st, sl) = spec.decode_step_logits(t).expect("spec continuation");
        let (pt, pl) = plain.decode_step_logits(t).expect("plain continuation");
        let diff = bit_diff(&sl, &pl);
        assert_eq!(
            diff, 0,
            "continuation step {i}: {diff} logits differ (spec {st} vs plain {pt}) after a \
             partial accept. Laguna's KV cache is linear and every row bounds its read by its own \
             total, so speculative rows past the accepted prefix must be invisible"
        );
    }
}

fn ckpt_dir() -> Option<std::path::PathBuf> {
    hub_snapshot::dir_from_env_or_hub(
        "NV_LAGUNA_DIR",
        "poolside/Laguna-XS-2.1-NVFP4",
        &["config.json", "*.safetensors"],
    )
}

#[test]
#[ignore = "loads the Laguna-XS-2.1-NVFP4 checkpoint; set NV_LAGUNA_WGPU_TEST=1"]
fn laguna_real_weights_verify_chain_matches_per_token_decode() {
    if std::env::var("NV_LAGUNA_WGPU_TEST").is_err() {
        eprintln!("[skip] NV_LAGUNA_WGPU_TEST not set");
        return;
    }
    if !have_gpu() {
        panic!("real-weights test needs a wgpu adapter");
    }
    let Some(dir) = ckpt_dir() else {
        hub_snapshot::precondition_absent(
            "laguna_real_weights_verify_chain_matches_per_token_decode",
            "no poolside/Laguna-XS-2.1-NVFP4 snapshot with safetensors",
            "set NV_LAGUNA_DIR to a Laguna-XS-2.1-NVFP4 snapshot dir, or cache the repo",
        );
        return;
    };
    let cfg = LagunaConfig::from_hf_json_file(&dir.join("config.json")).expect("parse config");
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &nv_weights::Device::Cpu).expect("open weights");
    let max_seq: usize = std::env::var("NV_LAGUNA_MAX_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let mut gpu = LagunaWgpu::from_loader(cfg, &loader, max_seq).expect("build from loader");
    let rows = gpu.verify_max_rows();
    assert!(rows >= 2, "verify entry absent on the real checkpoint");

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let text = std::env::var("NV_LAGUNA_PREFILL_TEXT").unwrap_or_else(|_| {
        "The verify forward must produce the same rows as one-at-a-time stepping. ".repeat(8)
    });
    let full: Vec<u32> = tok
        .encode(text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let k = rows.min(8);
    assert!(full.len() > k + 4, "corpus too short for a {k}-row chain");
    let prompt = &full[..full.len() - k];
    let chain = &full[full.len() - k..];

    gpu.reset().expect("reset");
    let done = gpu.prefill_tokens(prompt).expect("chunked prefill");
    for t in &prompt[done..] {
        gpu.prefill_step(*t).expect("tail");
    }
    let got = gpu.verify_chain(chain).expect("verify_chain");

    gpu.reset().expect("reset");
    let done = gpu.prefill_tokens(prompt).expect("chunked prefill");
    for t in &prompt[done..] {
        gpu.prefill_step(*t).expect("tail");
    }
    let want: Vec<u32> = chain
        .iter()
        .map(|t| gpu.decode_step(*t).expect("decode step"))
        .collect();
    assert_eq!(
        got, want,
        "real-weights verify_chain rows differ from the same tokens stepped one at a time"
    );
    eprintln!("[laguna-real-verify] k={k} rows={got:?}");
}

const LG4_SHIFT_DECODE_REBALANCE_IS_EXACT: &str =
    "lgw_gemv_nvfp4 rebalances the block-scale product across three exact power-of-two moves: \
     ws pure-shift decode lands true*2^-120 (exact for all 128 ue4m3 codes because f32 \
     subnormals cover the ue4m3 subnormal range), xs decode rebiased +2^24 lands bs at \
     true*2^-96 which is always a normal f32 (min true bs = 2^-9*2^-9 = 2^-18 -> 2^-114), and \
     the dot helper multiplies the packed-int8 dot by 0.25*2^96 (max |d| 18432 -> below 2^108, \
     finite). The fma therefore sees the same real product as the pre-rebalance kernel and \
     rounds identically, so the kernel output is bit-identical; this host replica pins each \
     move exhaustively";

fn ue4m3_select_decode(bits: u32) -> f32 {
    let b = bits & 127;
    if b < 8 {
        b as f32 * 0.001953125
    } else {
        f32::from_bits((b << 20) + 0x3c000000)
    }
}

fn ue4m3_shift_decode(bits: u32) -> f32 {
    f32::from_bits((bits & 127) << 20)
}

fn ue4m3_rebased_2pow24_decode(bits: u32) -> f32 {
    let b = bits & 127;
    if b < 8 {
        b as f32 * 32768.0
    } else {
        f32::from_bits((b << 20) + 0x48000000)
    }
}

#[test]
fn laguna_gemv_shift_decode_rebalance_is_bit_exact() {
    let two_pow_m120 = f64::from(nv_kernels::shift_decode_fold::E4M3_SHIFT_DECODE_LANDS_2POW120_BELOW_TRUE).recip();
    for code in 0u32..256 {
        let dec = f64::from(ue4m3_select_decode(code));
        assert_eq!(
            f64::from(ue4m3_shift_decode(code)),
            dec * two_pow_m120,
            "ws shift decode inexact at code {code}: {LG4_SHIFT_DECODE_REBALANCE_IS_EXACT}"
        );
        assert_eq!(
            f64::from(ue4m3_rebased_2pow24_decode(code)),
            dec * 16777216.0,
            "xs rebiased decode inexact at code {code}: {LG4_SHIFT_DECODE_REBALANCE_IS_EXACT}"
        );
    }
    let two_pow_m96 = (-96.0f64).exp2();
    for ws in 0u32..128 {
        for xs in 0u32..128 {
            let bs_old = ue4m3_select_decode(ws) * ue4m3_select_decode(xs);
            let bs_new = ue4m3_shift_decode(ws) * ue4m3_rebased_2pow24_decode(xs);
            assert_eq!(
                f64::from(bs_new),
                f64::from(bs_old) * two_pow_m96,
                "bs rebalance inexact at ws={ws} xs={xs}: {LG4_SHIFT_DECODE_REBALANCE_IS_EXACT}"
            );
            assert!(
                bs_new == 0.0 || bs_new.abs() >= f32::MIN_POSITIVE,
                "bs left the normal range at ws={ws} xs={xs}: {LG4_SHIFT_DECODE_REBALANCE_IS_EXACT}"
            );
        }
    }
    let quarter_2pow96 = 96.0f64.exp2() * 0.25;
    for d in -18432i32..=18432 {
        let dot_old = d as f32 * 0.25;
        let dot_new = d as f32 * 1.9807040628566084e28;
        assert_eq!(
            f64::from(dot_new),
            f64::from(d) * quarter_2pow96,
            "dot fold inexact at d={d}: {LG4_SHIFT_DECODE_REBALANCE_IS_EXACT}"
        );
        assert!(
            dot_new.is_finite() && f64::from(dot_new) == f64::from(dot_old) * 96.0f64.exp2(),
            "dot fold drifted from 2^96 times the old dot at d={d}"
        );
    }
}
