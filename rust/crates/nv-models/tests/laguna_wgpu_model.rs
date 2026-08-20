#![cfg(feature = "laguna-wip")]

mod common;
use common::CFG;
use nv_models::laguna::LagunaConfig;
use nv_models::laguna_wgpu::config::LagunaShapes;
use nv_models::laguna_wgpu::weights::{
    names, random_host_weights, HostExperts, HostFfn, HostLin, HostWeights,
};
use nv_models::laguna_wgpu::{ref_argmax, reference_step, LagunaWgpu, RefState};

const MAX_SEQ: usize = 24;
const TOKENS: [u32; 8] = [3, 17, 5, 42, 8, 61, 12, 30];

const BF16_ULP: f32 = 1.0 / 256.0;
const DEPTH_ULPS: f32 = 1.5;
const BOUND: f32 = DEPTH_ULPS * BF16_ULP;
const SENSITIVITY: f32 = 1e-1;

fn config_of(json: &str) -> LagunaConfig {
    LagunaConfig::from_hf_json_str(json).unwrap()
}

fn shapes_of(cfg: &LagunaConfig) -> LagunaShapes {
    LagunaShapes::derive(cfg, MAX_SEQ).unwrap()
}

fn have_gpu() -> bool {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(_) => true,
        Err(e) => {
            eprintln!("SKIP: no wgpu adapter: {e}");
            false
        }
    }
}

fn build(cfg: &LagunaConfig, hw: &HostWeights) -> LagunaWgpu {
    LagunaWgpu::new(cfg.clone(), hw, MAX_SEQ).unwrap()
}

fn scale_of(want: &[f32]) -> f32 {
    want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6)
}

fn worst_rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = scale_of(want);
    got.iter()
        .zip(want)
        .fold(0f32, |a, (g, w)| a.max((g - w).abs() / scale))
}

fn top2_gap(v: &[f32]) -> f32 {
    let mut s: Vec<f32> = v.to_vec();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    s[0] - s[1]
}

fn spread(v: &[f32]) -> f32 {
    let lo = v.iter().fold(f32::INFINITY, |a, x| a.min(*x));
    let hi = v.iter().fold(f32::NEG_INFINITY, |a, x| a.max(*x));
    hi - lo
}

struct Run {
    steps: usize,
    worst: f32,
    decided: usize,
    ties: usize,
    exact: usize,
    last: Vec<f32>,
}

fn drive(json: &str, seed: u64, steps: usize) -> Option<Run> {
    if !have_gpu() {
        return None;
    }
    let cfg = config_of(json);
    let shapes = shapes_of(&cfg);
    let hw = random_host_weights(&shapes, seed);
    let mut m = build(&cfg, &hw);
    let mut st = RefState::new(&shapes);

    let mut worst = 0f32;
    let mut decided = 0usize;
    let mut ties = 0usize;
    let mut exact = 0usize;
    let mut last = Vec::new();

    for i in 0..steps {
        let t = TOKENS[i % TOKENS.len()];
        let (gt, gl) = m.decode_step_logits(t).unwrap();
        let rl = reference_step(&shapes, &hw, &mut st, t).unwrap();
        assert_eq!(gl.len(), shapes.vocab_size);
        assert!(
            gl.iter().all(|v| v.is_finite()),
            "step {i}: gpu logits contain a non-finite value"
        );
        assert!(
            spread(&rl) > 1e-3,
            "step {i}: cpu oracle logits are degenerate (spread {})",
            spread(&rl)
        );
        let rel = worst_rel(&gl, &rl);
        worst = worst.max(rel);
        if gl.iter().zip(&rl).all(|(a, b)| a.to_bits() == b.to_bits()) {
            exact += 1;
        }

        assert_eq!(
            gt,
            ref_argmax(&rl),
            "step {i}: gpu argmax {gt} != oracle argmax {} (top-2 gap {:.4}, worst rel {rel:.3e})",
            ref_argmax(&rl),
            top2_gap(&rl)
        );
        if top2_gap(&rl) > 2.0 * BOUND * scale_of(&rl) {
            decided += 1;
        } else {
            ties += 1;
        }
        last = gl;
    }
    assert_eq!(m.current_pos(), steps);
    Some(Run {
        steps,
        worst,
        decided,
        ties,
        exact,
        last,
    })
}

fn check(label: &str, r: &Run) {
    eprintln!(
        "[laguna-wgpu] {label}: {} steps, worst rel {:.3e} ({:.2} bf16 ulp), {}/{} bit-identical, argmax {}/{}, {} decided, {} within-noise",
        r.steps,
        r.worst,
        r.worst / BF16_ULP,
        r.exact,
        r.steps,
        r.steps,
        r.steps,
        r.decided,
        r.ties
    );
    assert!(
        r.worst < BOUND,
        "{label}: worst relative logit error {:.3e} exceeds {DEPTH_ULPS} bf16 ulp ({BOUND:.3e})",
        r.worst
    );
}

#[test]
fn full_graph_matches_cpu_oracle_over_six_steps() {
    let Some(r) = drive(CFG, 0x51ee_d0d0_1234_beef, 6) else {
        return;
    };
    check("hybrid 4L", &r);
}

#[test]
fn graph_stays_locked_past_the_sliding_window() {
    let Some(r) = drive(CFG, 0x0bad_c0de_0000_0007, 16) else {
        return;
    };
    check("hybrid 4L past window", &r);
}

fn cfg_with_depth(n: usize) -> String {
    let kinds: Vec<&str> = (0..n)
        .map(|i| {
            if i % 4 == 0 || i % 4 == 3 {
                "\"full_attention\""
            } else {
                "\"sliding_attention\""
            }
        })
        .collect();
    let mlps: Vec<&str> = (0..n)
        .map(|i| {
            if i % 4 == 0 || i % 4 == 3 {
                "\"dense\""
            } else {
                "\"sparse\""
            }
        })
        .collect();
    let heads: Vec<&str> = (0..n)
        .map(|i| if i % 4 == 0 || i % 4 == 3 { "4" } else { "8" })
        .collect();
    CFG.replace(
        "\"num_hidden_layers\": 4",
        &format!("\"num_hidden_layers\": {n}"),
    )
    .replace(
        "\"layer_types\": [\"full_attention\", \"sliding_attention\", \"sliding_attention\", \"full_attention\"]",
        &format!("\"layer_types\": [{}]", kinds.join(", ")),
    )
    .replace(
        "\"mlp_layer_types\": [\"dense\", \"sparse\", \"sparse\", \"dense\"]",
        &format!("\"mlp_layer_types\": [{}]", mlps.join(", ")),
    )
    .replace(
        "\"num_attention_heads_per_layer\": [4, 8, 8, 4]",
        &format!("\"num_attention_heads_per_layer\": [{}]", heads.join(", ")),
    )
}

const SEEDS: [u64; 3] = [
    0x51ee_d0d0_1234_beef,
    0x51ee_d0d0_1234_affe,
    0x51ee_d0d0_1234_9ccd,
];

fn worst_over_seeds(json: &str) -> Option<f32> {
    let mut w = 0f32;
    for s in SEEDS {
        let r = drive(json, s, 6)?;
        w = w.max(r.worst);
    }
    Some(w)
}

#[test]
fn dense_stack_stays_bit_exact_as_depth_grows() {
    for n in [1usize, 2, 4, 8, 12] {
        let dense = cfg_with_depth(n).replace("\"sparse\"", "\"dense\"");
        let mut worst = 0f32;
        let mut exact = 0usize;
        let mut total = 0usize;
        for s in SEEDS {
            let Some(r) = drive(&dense, s, 6) else {
                return;
            };
            worst = worst.max(r.worst);
            exact += r.exact;
            total += r.steps;
        }
        eprintln!(
            "[laguna-wgpu] dense depth {n}: {exact}/{total} steps bit-identical over {} seeds, worst rel {worst:.3e} ({:.2} bf16 ulp)",
            SEEDS.len(),
            worst / BF16_ULP
        );
        assert_eq!(
            exact, total,
            "depth {n}: the order-replicating oracle should reproduce the dense stack bit-for-bit (worst rel {worst:.3e})"
        );
    }
}

#[test]
fn deep_moe_routing_agrees_with_the_oracle_and_topk_is_load_bearing() {
    let seed = SEEDS[0];
    let moe8 = cfg_with_depth(8);
    let Some(topk2) = drive(&moe8, seed, 6) else {
        return;
    };
    let all_experts = moe8.replace("\"num_experts_per_tok\": 2", "\"num_experts_per_tok\": 4");
    let Some(topk_all) = drive(&all_experts, seed, 6) else {
        return;
    };
    eprintln!(
        "[laguna-wgpu] moe depth 8 seed {seed:#x}: top-k=2 {:.3e} ({}/{} exact) -> top-k=all {:.3e} ({}/{} exact)",
        topk2.worst,
        topk2.exact,
        topk2.steps,
        topk_all.worst,
        topk_all.exact,
        topk_all.steps
    );
    assert_eq!(
        topk2.exact, topk2.steps,
        "the gpu router selected a different expert set than the oracle at depth 8 (worst rel {:.3e})",
        topk2.worst
    );
    assert_eq!(
        topk_all.exact, topk_all.steps,
        "selecting every expert still diverged (worst rel {:.3e})",
        topk_all.worst
    );
    let scale = topk2
        .last
        .iter()
        .fold(0f32, |a, v| a.max(v.abs()))
        .max(1e-6);
    let d = topk2
        .last
        .iter()
        .zip(&topk_all.last)
        .fold(0f32, |a, (x, y)| a.max((x - y).abs()))
        / scale;
    assert!(
        d > SENSITIVITY,
        "widening top-k from 2 to every expert moved the logits by only {d:.3e}; the routing decision is not load-bearing"
    );
}

#[test]
fn shallow_moe_tracks_the_dense_budget_across_seeds() {
    let Some(w) = worst_over_seeds(&cfg_with_depth(4)) else {
        return;
    };
    eprintln!(
        "[laguna-wgpu] moe depth 4: worst rel over {} seeds {:.3e} ({:.2} bf16 ulp)",
        SEEDS.len(),
        w,
        w / BF16_ULP
    );
    assert!(
        w < BOUND,
        "4-layer MoE disagreed by {:.2} ulp across seeds, past the {DEPTH_ULPS} ulp budget",
        w / BF16_ULP
    );
}

#[test]
fn state_carries_across_steps_and_reset_is_exact() {
    if !have_gpu() {
        return;
    }
    let cfg = config_of(CFG);
    let shapes = shapes_of(&cfg);
    let hw = random_host_weights(&shapes, 11);
    let mut m = build(&cfg, &hw);

    let (_, fresh1) = m.decode_step_logits(TOKENS[1]).unwrap();
    m.reset().unwrap();
    assert_eq!(m.current_pos(), 0);

    let (_, l0) = m.decode_step_logits(TOKENS[0]).unwrap();
    let (_, carried1) = m.decode_step_logits(TOKENS[1]).unwrap();
    assert!(
        worst_rel(&carried1, &fresh1) > 1e-2,
        "token {} at position 1 reproduced its position-0 logits; the kv cache is not carrying",
        TOKENS[1]
    );

    m.reset().unwrap();
    let (_, l0b) = m.decode_step_logits(TOKENS[0]).unwrap();
    assert_eq!(
        l0b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        l0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "reset+replay did not reproduce step 0 bit-for-bit"
    );
}

#[test]
fn tied_embeddings_use_the_embedding_matrix_as_the_head() {
    let tied = CFG.replace(
        "\"tie_word_embeddings\": false",
        "\"tie_word_embeddings\": true",
    );
    assert!(tied.contains("\"tie_word_embeddings\": true"));
    let Some(r) = drive(&tied, 0x71ed_0000_0000_0001, 4) else {
        return;
    };
    check("tied head", &r);

    let cfg = config_of(&tied);
    let shapes = shapes_of(&cfg);
    assert!(shapes.tie_word_embeddings);
    let hw = random_host_weights(&shapes, 3);
    assert_eq!(hw.lm_head, hw.embed);
}

#[test]
fn ungated_variant_runs_the_same_graph() {
    let ungated = CFG.replace("\"gating\": \"per-head\"", "\"gating\": false");
    let Some(r) = drive(&ungated, 0x9999_0000_1111_2222, 5) else {
        return;
    };
    check("ungated", &r);
}

#[test]
fn all_dense_and_all_moe_variants_both_assemble() {
    for (name, json) in [
        (
            "all-dense",
            CFG.replace(
                "\"mlp_layer_types\": [\"dense\", \"sparse\", \"sparse\", \"dense\"]",
                "\"mlp_layer_types\": [\"dense\", \"dense\", \"dense\", \"dense\"]",
            ),
        ),
        (
            "all-moe",
            CFG.replace(
                "\"mlp_layer_types\": [\"dense\", \"sparse\", \"sparse\", \"dense\"]",
                "\"mlp_layer_types\": [\"sparse\", \"sparse\", \"sparse\", \"sparse\"]",
            ),
        ),
    ] {
        let Some(r) = drive(&json, 0x4242_0000_0000_0001, 4) else {
            return;
        };
        check(name, &r);
    }
}

#[test]
fn oracle_is_sensitive_to_a_perturbed_lm_head() {
    if !have_gpu() {
        return;
    }
    let cfg = config_of(CFG);
    let shapes = shapes_of(&cfg);
    let hw = random_host_weights(&shapes, 5);
    let mut m = build(&cfg, &hw);
    let (_, gl) = m.decode_step_logits(TOKENS[0]).unwrap();

    let mut bad = hw.clone();
    for v in bad.lm_head.iter_mut().take(shapes.hidden_size) {
        *v ^= 0x0080;
    }
    let mut clean = RefState::new(&shapes);
    let good = reference_step(&shapes, &hw, &mut clean, TOKENS[0]).unwrap();
    assert!(
        gl.iter()
            .zip(&good)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the unperturbed oracle is not bit-identical; the sensitivity margin is unanchored"
    );
    let mut st = RefState::new(&shapes);
    let rl = reference_step(&shapes, &bad, &mut st, TOKENS[0]).unwrap();
    assert!(
        worst_rel(&gl, &rl) > SENSITIVITY,
        "perturbing lm_head row 0 moved the oracle by only {:.3e}; the comparison is not load-bearing",
        worst_rel(&gl, &rl)
    );
}

#[test]
fn oracle_is_sensitive_to_a_perturbed_middle_layer() {
    if !have_gpu() {
        return;
    }
    let cfg = config_of(CFG);
    let shapes = shapes_of(&cfg);
    let hw = random_host_weights(&shapes, 5);
    let mut m = build(&cfg, &hw);
    let (_, gl) = m.decode_step_logits(TOKENS[0]).unwrap();

    let mut bad = hw.clone();
    for v in bad.layers[2].post_attn_ln.iter_mut() {
        *v ^= 0x0040;
    }
    let mut clean = RefState::new(&shapes);
    let good = reference_step(&shapes, &hw, &mut clean, TOKENS[0]).unwrap();
    assert!(
        gl.iter()
            .zip(&good)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the unperturbed oracle is not bit-identical; the sensitivity margin is unanchored"
    );
    let mut st = RefState::new(&shapes);
    let rl = reference_step(&shapes, &bad, &mut st, TOKENS[0]).unwrap();
    assert!(
        worst_rel(&gl, &rl) > SENSITIVITY,
        "perturbing layer 2's post-attention norm moved the oracle by only {:.3e}",
        worst_rel(&gl, &rl)
    );
}

#[test]
fn decode_rejects_out_of_range_tokens_and_cache_overflow() {
    if !have_gpu() {
        return;
    }
    let cfg = config_of(CFG);
    let shapes = shapes_of(&cfg);
    let hw = random_host_weights(&shapes, 2);
    let mut m = build(&cfg, &hw);
    assert!(m.decode_step(shapes.vocab_size as u32).is_err());
    m.reset().unwrap();
    for _ in 0..MAX_SEQ {
        m.decode_step(1).unwrap();
    }
    assert!(
        m.decode_step(1).is_err(),
        "decode past max_seq_tokens must fail rather than corrupt the cache"
    );
}

enum St {
    Bf16(Vec<u16>),
    F32(Vec<f32>),
}

fn st_bytes(t: &St) -> Vec<u8> {
    match t {
        St::Bf16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        St::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

fn st_dtype(t: &St) -> &'static str {
    match t {
        St::Bf16(_) => "BF16",
        St::F32(_) => "F32",
    }
}

fn write_safetensors(path: &std::path::Path, items: &[(String, St, Vec<usize>)]) {
    let mut blob: Vec<u8> = Vec::new();
    let mut entries: Vec<String> = Vec::new();
    for (name, t, shape) in items {
        let b = st_bytes(t);
        let start = blob.len();
        blob.extend_from_slice(&b);
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        entries.push(format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            name,
            st_dtype(t),
            dims.join(","),
            start,
            blob.len()
        ));
    }
    let mut header = format!("{{{}}}", entries.join(","));
    while (header.len() + 8) % 8 != 0 {
        header.push(' ');
    }
    let mut out: Vec<u8> = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&blob);
    std::fs::write(path, out).unwrap();
}

fn push_lin(items: &mut Vec<(String, St, Vec<usize>)>, module: &str, l: &HostLin) {
    match l {
        HostLin::Bf16(b) => items.push((
            names::bf16_weight(module),
            St::Bf16(b.w.clone()),
            vec![b.n, b.k],
        )),
        HostLin::Nvfp4(_) => panic!("loader fixture is bf16 only"),
    }
}

fn serialize_host_weights(
    shapes: &LagunaShapes,
    hw: &HostWeights,
) -> Vec<(String, St, Vec<usize>)> {
    let hidden = shapes.hidden_size;
    let mut items: Vec<(String, St, Vec<usize>)> = vec![
        (
            names::EMBED.to_string(),
            St::Bf16(hw.embed.clone()),
            vec![shapes.vocab_size, hidden],
        ),
        (
            names::FINAL_NORM.to_string(),
            St::Bf16(hw.final_norm.clone()),
            vec![hidden],
        ),
        (
            names::LM_HEAD.to_string(),
            St::Bf16(hw.lm_head.clone()),
            vec![shapes.vocab_size, hidden],
        ),
    ];
    for (li, layer) in hw.layers.iter().enumerate() {
        items.push((
            names::input_layernorm(li),
            St::Bf16(layer.input_ln.clone()),
            vec![hidden],
        ));
        items.push((
            names::post_attention_layernorm(li),
            St::Bf16(layer.post_attn_ln.clone()),
            vec![hidden],
        ));
        push_lin(&mut items, &names::q_proj(li), &layer.attn.q);
        push_lin(&mut items, &names::k_proj(li), &layer.attn.k);
        push_lin(&mut items, &names::v_proj(li), &layer.attn.v);
        push_lin(&mut items, &names::o_proj(li), &layer.attn.o);
        if let Some(g) = &layer.attn.g {
            push_lin(&mut items, &names::g_proj(li), g);
        }
        items.push((
            names::q_norm(li),
            St::Bf16(layer.attn.q_norm.clone()),
            vec![shapes.head_dim],
        ));
        items.push((
            names::k_norm(li),
            St::Bf16(layer.attn.k_norm.clone()),
            vec![shapes.head_dim],
        ));
        match &layer.ffn {
            HostFfn::Dense(d) => {
                push_lin(&mut items, &names::dense_gate_proj(li), &d.gate);
                push_lin(&mut items, &names::dense_up_proj(li), &d.up);
                push_lin(&mut items, &names::dense_down_proj(li), &d.down);
            }
            HostFfn::Moe(m) => {
                items.push((
                    names::router(li),
                    St::Bf16(m.router.w.clone()),
                    vec![m.router.n, m.router.k],
                ));
                items.push((
                    names::selection_bias(li),
                    St::F32(m.selection_bias.clone()),
                    vec![shapes.num_experts],
                ));
                for e in 0..shapes.num_experts {
                    let (g, u, d) = match (&m.experts_gate, &m.experts_up, &m.experts_down) {
                        (HostExperts::Bf16(_), HostExperts::Bf16(_), HostExperts::Bf16(_)) => (
                            m.experts_gate.expert(e),
                            m.experts_up.expert(e),
                            m.experts_down.expert(e),
                        ),
                        _ => panic!("loader fixture is bf16 only"),
                    };
                    push_lin(&mut items, &names::expert_gate_proj(li, e), &g);
                    push_lin(&mut items, &names::expert_up_proj(li, e), &u);
                    push_lin(&mut items, &names::expert_down_proj(li, e), &d);
                }
                push_lin(&mut items, &names::shared_gate_proj(li), &m.shared_gate);
                push_lin(&mut items, &names::shared_up_proj(li), &m.shared_up);
                push_lin(&mut items, &names::shared_down_proj(li), &m.shared_down);
            }
        }
    }
    items
}

#[test]
fn from_loader_reproduces_the_host_graph_bit_for_bit() {
    if !have_gpu() {
        return;
    }
    let cfg = config_of(CFG);
    let shapes = shapes_of(&cfg);
    let hw = random_host_weights(&shapes, 0xfeed_face_0000_0001);

    let dir = std::env::temp_dir().join(format!("laguna-wgpu-loader-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    write_safetensors(&path, &serialize_host_weights(&shapes, &hw));

    let loader = nv_weights::WeightLoader::open_file(&path, &nv_weights::Device::Cpu).unwrap();
    for n in [names::EMBED, names::LM_HEAD, names::FINAL_NORM] {
        assert!(loader.has(n), "fixture is missing {n}");
    }

    let mut from_host = build(&cfg, &hw);
    let mut from_disk = LagunaWgpu::from_loader(cfg.clone(), &loader, MAX_SEQ).unwrap();
    assert_eq!(from_host.pass_count(), from_disk.pass_count());

    for i in 0..4 {
        let t = TOKENS[i];
        let (ha, hl) = from_host.decode_step_logits(t).unwrap();
        let (da, dl) = from_disk.decode_step_logits(t).unwrap();
        assert_eq!(ha, da, "step {i}: loader graph picked a different token");
        assert_eq!(
            hl.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            dl.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "step {i}: loader graph logits are not bit-identical to the host graph"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn from_loader_falls_back_to_the_embedding_when_the_head_is_tied() {
    if !have_gpu() {
        return;
    }
    let tied = CFG.replace(
        "\"tie_word_embeddings\": false",
        "\"tie_word_embeddings\": true",
    );
    let cfg = config_of(&tied);
    let shapes = shapes_of(&cfg);
    let mut hw = random_host_weights(&shapes, 0xfeed_face_0000_0002);
    hw.lm_head = hw.embed.clone();

    let dir = std::env::temp_dir().join(format!("laguna-wgpu-tied-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let mut items = serialize_host_weights(&shapes, &hw);
    items.retain(|(n, _, _)| n != names::LM_HEAD);
    write_safetensors(&path, &items);

    let loader = nv_weights::WeightLoader::open_file(&path, &nv_weights::Device::Cpu).unwrap();
    assert!(!loader.has(names::LM_HEAD));

    let mut from_host = build(&cfg, &hw);
    let mut from_disk = LagunaWgpu::from_loader(cfg.clone(), &loader, MAX_SEQ).unwrap();
    let (ha, hl) = from_host.decode_step_logits(TOKENS[0]).unwrap();
    let (da, dl) = from_disk.decode_step_logits(TOKENS[0]).unwrap();
    assert_eq!(ha, da);
    assert_eq!(
        hl.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        dl.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "tied head did not fall back to the embedding matrix"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn from_loader_reports_a_missing_tensor_by_name() {
    let cfg = config_of(CFG);
    let shapes = shapes_of(&cfg);
    let hw = random_host_weights(&shapes, 9);

    let dir = std::env::temp_dir().join(format!("laguna-wgpu-missing-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    let mut items = serialize_host_weights(&shapes, &hw);
    let dropped = names::k_norm(2);
    items.retain(|(n, _, _)| *n != dropped);
    write_safetensors(&path, &items);

    let loader = nv_weights::WeightLoader::open_file(&path, &nv_weights::Device::Cpu).unwrap();
    let msg = match LagunaWgpu::from_loader(cfg, &loader, MAX_SEQ) {
        Ok(_) => panic!("a missing k_norm must not build a graph"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        msg.contains(&dropped),
        "error did not name the missing tensor: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn graph_shape_is_reported() {
    if !have_gpu() {
        return;
    }
    let cfg = config_of(CFG);
    let shapes = shapes_of(&cfg);
    let hw = random_host_weights(&shapes, 1);
    let m = build(&cfg, &hw);
    assert_eq!(m.max_seq_tokens(), MAX_SEQ);
    assert_eq!(m.config().num_hidden_layers, 4);
    assert_eq!(m.shapes().num_layers, 4);
    assert!(m.pass_count() > shapes.num_layers * 8);
    assert!(m.vram_report().total_bytes > 0);
    assert_eq!(m.current_pos(), 0);
}
