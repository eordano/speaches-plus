#![cfg(feature = "wgpu")]

mod common;
use common::distinct;
use common::envn;
use common::LcgCentered0p1Shift32 as Lcg;
use std::time::Instant;

use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_wgpu::{
    quantize_nvfp4_host, Gemma4Wgpu, HostBf16Lin, HostLayer, HostProj, HostWeights,
};
use common::config_json_gemma4_layers as config_json;

const CONTINUATION_TOKENS: usize = 8;

fn gpu_or_refuse() {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => eprintln!("[g4-pf4] adapter: {}", ctx.summary()),
        Err(e) => panic!("gemma4 nvfp4 chunked prefill needs a wgpu adapter: {e}"),
    }
}

fn host_weights_with_nvfp4_ffn_and_o(config: &Gemma4Config, seed: u64) -> HostWeights {
    let mut rng = Lcg(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let n_q = config.num_attention_heads;
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        let kind = config.layer_kind(i);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_rows = q_dim + kv_dim * if has_v { 2 } else { 1 };
        let bf16 = |rng: &mut Lcg, n: usize, k: usize| {
            HostProj::Bf16(HostBf16Lin {
                w: rng.bf16_vec(n * k),
                n,
                k,
            })
        };
        let quant = |rng: &mut Lcg, n: usize, k: usize| {
            HostProj::Nvfp4(quantize_nvfp4_host(&rng.bf16_vec(n * k), n, k))
        };
        layers.push(HostLayer {
            kind,
            input_ln: rng.bf16_vec_around_one(hidden),
            post_attn_ln: rng.bf16_vec_around_one(hidden),
            pre_ff_ln: rng.bf16_vec_around_one(hidden),
            post_ff_ln: rng.bf16_vec_around_one(hidden),
            q_norm: rng.bf16_vec_around_one(hd),
            k_norm: rng.bf16_vec_around_one(hd),
            layer_scalar: 0.9,
            has_v,
            qkv: bf16(&mut rng, qkv_rows, hidden),
            o: quant(&mut rng, hidden, q_dim),
            gate_up: quant(&mut rng, 2 * inter, hidden),
            down: quant(&mut rng, hidden, inter),
        });
    }
    HostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

struct Arm {
    continuation: Vec<u32>,
    logit_bits: Vec<u32>,
}

fn greedy_after_prefill(m: &mut Gemma4Wgpu, ids: &[u32], chunked: bool) -> Arm {
    m.reset();
    let (last, rest) = ids.split_last().expect("prompt");
    let mut done = 0usize;
    if chunked {
        done = m.prefill_tokens(rest).expect("prefill_tokens");
        assert!(
            done > 0 || rest.len() < m.prefill_chunk_len(),
            "chunked prefill consumed nothing on a {}-token prompt at m={}",
            rest.len(),
            m.prefill_chunk_len()
        );
    }
    for t in &rest[done..] {
        m.prefill_step(*t).expect("prefill step");
    }
    let (mut next, logits) = m.decode_step_logits(*last).expect("last prompt token");
    let mut continuation = Vec::with_capacity(CONTINUATION_TOKENS);
    for _ in 0..CONTINUATION_TOKENS {
        continuation.push(next);
        next = m.decode_step(next).expect("decode step");
    }
    Arm {
        continuation,
        logit_bits: logits.into_iter().map(f32::to_bits).collect(),
    }
}

fn run_prompt_lengths(m: &mut Gemma4Wgpu, vocab: usize, arm_name: &str) {
    let cm = m.prefill_chunk_len();
    for pp in [cm * 2 + cm / 2 + 1, cm + 3, cm.max(3) - 1] {
        if pp < 2 {
            continue;
        }
        let ids: Vec<u32> = (0..pp).map(|i| ((i * 7919 + 13) % vocab) as u32).collect();
        let chunked = greedy_after_prefill(m, &ids, true);
        let replay = greedy_after_prefill(m, &ids, false);
        let d = distinct(&replay.logit_bits);
        assert!(
            d > (vocab / 4).min(1000),
            "{arm_name} pp={pp}: logits are degenerate ({d} distinct of {vocab}); the bit-compare would be vacuous"
        );
        let diff = chunked
            .logit_bits
            .iter()
            .zip(replay.logit_bits.iter())
            .filter(|(a, b)| a != b)
            .count();
        let worst = chunked
            .logit_bits
            .iter()
            .zip(replay.logit_bits.iter())
            .map(|(a, b)| (f32::from_bits(*a) - f32::from_bits(*b)).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[g4-pf4] {arm_name} pp={pp:3}: {diff}/{vocab} logit lanes differ, max |delta| {worst:.3e}; chunked {:?} replay {:?}",
            chunked.continuation, replay.continuation
        );
        assert_eq!(
            chunked.continuation, replay.continuation,
            "{arm_name} pp={pp}: chunked prefill and per-token replay produced different greedy continuations"
        );
        assert_eq!(
            diff, 0,
            "{arm_name} pp={pp}: {diff} of {vocab} logit lanes differ between chunked and per-token \
             nvfp4 prefill (max |delta| {worst:.3e}). The in-tree routing gate \
             (wgpu_gemma4_nvfp4_v2_routing.rs) already pins tree, sg and v2 decode kernels \
             bit-equal on nvfp4 operands, so the slot-strided M-row arm must match too; a drift \
             here means a slot stride or scale index defect, not acceptable float reassociation."
        );
    }
}

fn build_nvfp4_model(seed: u64, batch_slots: usize) -> (Gemma4Wgpu, usize) {
    let layers = envn("NV_G4_PF4_LAYERS", 4);
    let hidden = envn("NV_G4_PF4_HIDDEN", 512);
    let inter = envn("NV_G4_PF4_INTER", 1024);
    let vocab = envn("NV_G4_PF4_VOCAB", 2048);
    let raw = config_json(layers, hidden, inter, vocab);
    let config = Gemma4Config::from_hf_json_str(&raw).expect("config");
    let w = host_weights_with_nvfp4_ffn_and_o(&config, seed);
    let t = Instant::now();
    let m = if batch_slots >= 2 {
        Gemma4Wgpu::new_batched(config, &w, 512, batch_slots).expect("build")
    } else {
        Gemma4Wgpu::new(config, &w, 512).expect("build")
    };
    eprintln!(
        "[g4-pf4] built in {:.2}s: chunk m={}, {} prefill passes/chunk, nvfp4 v2 {:?}",
        t.elapsed().as_secs_f64(),
        m.prefill_chunk_len(),
        m.prefill_pass_count(),
        m.nvfp4_v2_projections(),
    );
    (m, vocab)
}

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn with_env<T>(env: &[(&str, &str)], f: impl FnOnce() -> T) -> T {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for (k, v) in env {
        std::env::set_var(k, v);
    }
    let out = f();
    for (k, _) in env {
        std::env::remove_var(k);
    }
    out
}

const W8_FFN_OFF_SO_NVFP4_SURVIVES_TO_THE_GRAPH: (&str, &str) = ("NV_G4_WGPU_W8_FFN", "off");
const O_PROJ_ALSO_NVFP4: (&str, &str) = ("NV_WGPU_O_NVFP4", "1");

#[test]
fn nvfp4_chunked_prefill_engages_and_reproduces_the_per_token_replay() {
    gpu_or_refuse();
    with_env(
        &[W8_FFN_OFF_SO_NVFP4_SURVIVES_TO_THE_GRAPH, O_PROJ_ALSO_NVFP4],
        || {
            let (mut m, vocab) = build_nvfp4_model(0x9e3779b9, 0);
            let (_, nvfp4_total) = m.nvfp4_v2_projections();
            assert!(
                nvfp4_total > 0,
                "no nvfp4 projection survived to the graph; this gate would cover nothing \
                 (NV_G4_WGPU_W8_FFN=off did not stick)"
            );
            assert!(
                m.prefill_chunk_len() >= 2,
                "an nvfp4 projection survived and chunked prefill is OFF: the slot-strided \
                 M-row arm regressed to the old blanket disable"
            );
            run_prompt_lengths(&mut m, vocab, "slot-tree");
        },
    );
}

#[test]
fn nvfp4_chunked_prefill_matches_replay_on_the_v2_subgroup_route_too() {
    gpu_or_refuse();
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared().expect("adapter");
    if !nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2::subgroup32_ok(ctx) {
        eprintln!("[g4-pf4] SKIP v2 arm: adapter subgroup width is not 32");
        return;
    }
    with_env(
        &[
            W8_FFN_OFF_SO_NVFP4_SURVIVES_TO_THE_GRAPH,
            O_PROJ_ALSO_NVFP4,
            ("NV_WGPU_NVFP4_V2", "1"),
        ],
        || {
            let (mut m, vocab) = build_nvfp4_model(0x9e3779b9, 0);
            let (v2_routed, nvfp4_total) = m.nvfp4_v2_projections();
            assert!(
                nvfp4_total > 0 && v2_routed > 0,
                "v2 arm routed {v2_routed} of {nvfp4_total} nvfp4 projections; the v2 slot \
                 kernels would go unexercised"
            );
            assert!(m.prefill_chunk_len() >= 2, "chunked prefill off on the v2 arm");
            run_prompt_lengths(&mut m, vocab, "slot-v2");
        },
    );
}

#[test]
fn nvfp4_batch_decode_stays_enabled_and_each_slot_matches_single_stream() {
    gpu_or_refuse();
    with_env(
        &[W8_FFN_OFF_SO_NVFP4_SURVIVES_TO_THE_GRAPH, O_PROJ_ALSO_NVFP4],
        || {
            let (mut m, vocab) = build_nvfp4_model(0x9e3779b9, 4);
            let (_, nvfp4_total) = m.nvfp4_v2_projections();
            assert!(nvfp4_total > 0, "no nvfp4 projection survived to the graph");
            assert_eq!(
                m.batch_slots(),
                4,
                "an nvfp4 projection survived and batch decode is OFF: the slot-strided \
                 M-row arm regressed to the old blanket disable"
            );
            let slots = m.batch_slots();
            let steps = CONTINUATION_TOKENS;
            let prompts: Vec<Vec<u32>> = (0..slots)
                .map(|j| {
                    (0..6)
                        .map(|i| (((i + 1) * 7919 + j * 613 + 13) % vocab) as u32)
                        .collect()
                })
                .collect();
            let mut solo: Vec<(Vec<u32>, Vec<Vec<u32>>)> = Vec::new();
            for p in &prompts {
                m.reset_slot(0).expect("reset slot 0");
                let (last, rest) = p.split_last().expect("prompt");
                for t in rest {
                    m.prefill_step(*t).expect("prefill step");
                }
                let mut toks = Vec::new();
                let mut bits = Vec::new();
                let mut t = *last;
                for _ in 0..steps {
                    let (n, lg) = m.decode_step_logits(t).expect("decode step logits");
                    bits.push(lg.into_iter().map(f32::to_bits).collect::<Vec<u32>>());
                    toks.push(n);
                    t = n;
                }
                solo.push((toks, bits));
            }
            let mut cur: Vec<u32> = Vec::with_capacity(slots);
            for (j, p) in prompts.iter().enumerate() {
                cur.push(m.prefill_slot(j, p).expect("prefill slot"));
            }
            for (j, t) in cur.iter().enumerate() {
                assert_eq!(
                    *t, solo[j].0[0],
                    "slot {j}: batched nvfp4 prefill already disagrees with the solo run"
                );
            }
            for i in 1..steps {
                let nx = m.decode_step_batch(&cur).expect("decode_step_batch");
                assert_eq!(nx.len(), slots);
                let lg = m.batch_logits().expect("batch logits");
                for j in 0..slots {
                    let want = &solo[j].1[i];
                    let got: Vec<u32> = lg[j * vocab..(j + 1) * vocab]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect();
                    let diff = want.iter().zip(got.iter()).filter(|(x, y)| x != y).count();
                    assert_eq!(
                        diff, 0,
                        "batch step {i} slot {j}: {diff} of {vocab} logit lanes differ from the \
                         solo nvfp4 run through the same graph"
                    );
                    assert_eq!(nx[j], solo[j].0[i], "batch step {i} slot {j}: token differs");
                }
                cur = nx;
            }
            eprintln!("[g4-pf4] batch decode: {slots} slots x {steps} steps bit-identical to solo");
        },
    );
}
