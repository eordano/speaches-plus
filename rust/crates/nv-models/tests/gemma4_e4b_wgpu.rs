#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::LcgCentered0p1Shift33 as Lcg;
use common::require;
use common::TINY_E4B_CONFIG;
mod official_template;

use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_e4b_wgpu::{
    e4b_host_weights_from_loader, E4bHostLayer, E4bHostWeights, Gemma4E4bWgpu, HostLin,
};
use official_template::OfficialTemplate;
use common::e4b_snapshot_dir;

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn tiny_e4b_host_weights(config: &Gemma4Config, seed: u64) -> E4bHostWeights {
    let mut rng = Lcg(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let hpl = config.hidden_size_per_layer_input;
    let n_q = config.num_attention_heads;
    let n_layers = config.num_hidden_layers;

    let mut layers = Vec::new();
    for i in 0..n_layers {
        let kind = config.layer_kind(i);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let kv_source = config.kv_source_layer(i);
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_rows = match kv_source {
            Some(_) => q_dim,
            None => q_dim + kv_dim * if has_v { 2 } else { 1 },
        };
        let k_norm = match kv_source {
            Some(_) => Vec::new(),
            None => rng.bf16_vec_around_one(hd),
        };
        layers.push(E4bHostLayer {
            kind,
            kv_source,
            input_ln: rng.bf16_vec_around_one(hidden),
            post_attn_ln: rng.bf16_vec_around_one(hidden),
            pre_ff_ln: rng.bf16_vec_around_one(hidden),
            post_ff_ln: rng.bf16_vec_around_one(hidden),
            post_per_layer_input_norm: rng.bf16_vec_around_one(hidden),
            q_norm: rng.bf16_vec_around_one(hd),
            k_norm,
            layer_scalar: 0.9,
            has_v,
            qkv: HostLin::new(rng.bf16_vec(qkv_rows * hidden), qkv_rows, hidden),
            o: HostLin::new(rng.bf16_vec(hidden * q_dim), hidden, q_dim),
            gate_up: HostLin::new(rng.bf16_vec(2 * inter * hidden), 2 * inter, hidden),
            down: HostLin::new(rng.bf16_vec(hidden * inter), hidden, inter),
            per_layer_input_gate: HostLin::new(rng.bf16_vec(hpl * hidden), hpl, hidden),
            per_layer_projection: HostLin::new(rng.bf16_vec(hidden * hpl), hidden, hpl),
        });
    }

    let ple_row = n_layers * hpl;
    E4bHostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        embed_per_layer: rng.bf16_vec(config.vocab_size_per_layer() * ple_row),
        per_layer_model_projection: HostLin::new(rng.bf16_vec(ple_row * hidden), ple_row, hidden),
        per_layer_projection_norm: rng.bf16_vec_around_one(hpl),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

#[test]
fn tiny_config_exercises_per_layer_embeddings_and_kv_sharing() {
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    assert!(config.has_per_layer_embeddings());
    assert_eq!(config.hidden_size_per_layer_input, 32);
    assert_eq!(config.first_kv_shared_layer_idx(), 4);
    let sources: Vec<Option<usize>> = (0..config.num_hidden_layers)
        .map(|i| config.kv_source_layer(i))
        .collect();
    eprintln!("kv sources: {sources:?}");
    assert_eq!(
        sources,
        vec![None, None, None, None, Some(2), Some(3)],
        "layer 4 (sliding) must borrow KV from 2, layer 5 (full) from 3"
    );
    let kinds: Vec<LayerType> = config.layer_types.clone();
    assert!(
        kinds.contains(&LayerType::FullAttention) && kinds.contains(&LayerType::SlidingAttention)
    );
}

#[test]
fn chunk_planner_keeps_every_chunk_even_and_within_the_binding_limit() {
    use nv_models::gemma4_e4b_wgpu::{plan_chunks, MAX_TABLE_CHUNKS};
    let cases = [
        (262144usize, 5120usize, 1u64 << 32),
        (262144, 21504, 1u64 << 32),
        (512, 384, 1u64 << 32),
        (131072, 21504, 1u64 << 32),
    ];
    for (rows, row_bytes, limit) in cases {
        let p = plan_chunks(rows, row_bytes, limit).unwrap();
        eprintln!("rows {rows} row_bytes {row_bytes} limit {limit} -> {p:?}");
        assert_eq!(p.rows_per_chunk % 2, 0, "chunk stride must stay even");
        assert!(p.n_chunks <= MAX_TABLE_CHUNKS);
        assert!(
            p.rows_per_chunk * p.n_chunks >= rows,
            "chunks must cover the table"
        );
        assert!(
            (p.rows_per_chunk * row_bytes) as u64 <= limit.min(1 << 30),
            "chunk exceeds the storage binding limit"
        );
        let last = rows - (p.n_chunks - 1) * p.rows_per_chunk;
        assert_eq!(last % 2, 0, "last chunk must have an even row count");
    }
    assert!(
        plan_chunks(262144, 1 << 28, 1 << 32).is_err(),
        "a table needing more than {MAX_TABLE_CHUNKS} chunks must be rejected, not silently truncated"
    );
}

#[test]
fn synthetic_e4b_decode_shapes_finiteness_determinism() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let weights = tiny_e4b_host_weights(&config, 0x5eed);
    let steps: Vec<u32> = vec![7, 300, 42, 511, 0, 13, 99, 250, 8, 9, 10, 11];

    let run = |label: &str| -> Vec<(u32, Vec<f32>)> {
        let guard = env_guard();
        let mut m = Gemma4E4bWgpu::new(
            Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
            &weights,
            64,
        )
        .unwrap();
        drop(guard);
        eprintln!(
            "[{label}] passes per decode step: {} weight bytes/token {}",
            m.pass_count(),
            m.weight_bytes_per_token()
        );
        steps
            .iter()
            .map(|t| m.decode_step_logits(*t).unwrap())
            .collect()
    };

    let a = run("run-a");
    for (i, (tok, logits)) in a.iter().enumerate() {
        assert_eq!(logits.len(), config.vocab_size);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "step {i}: non-finite logits"
        );
        let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mn = logits.iter().cloned().fold(f32::MAX, f32::min);
        assert!((*tok as usize) < config.vocab_size);
        assert!(
            mx > mn && mx.abs() <= 30.0 && mn.abs() <= 30.0,
            "step {i}: softcapped logits out of range: min {mn} max {mx}"
        );
        eprintln!("step {i}: argmax {tok} logit range [{mn:.4}, {mx:.4}]");
    }

    let b = run("run-b");
    let mut diff_words = 0usize;
    for ((_, la), (_, lb)) in a.iter().zip(b.iter()) {
        for (x, y) in la.iter().zip(lb.iter()) {
            if x.to_bits() != y.to_bits() {
                diff_words += 1;
            }
        }
    }
    assert_eq!(
        diff_words, 0,
        "wgpu E4B decode must be deterministic run-to-run"
    );
    eprintln!(
        "determinism: {} logits x {} steps bit-identical across two runs",
        config.vocab_size,
        steps.len()
    );
}

#[test]
fn synthetic_decode_chain_matches_stepped_bitwise() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let weights = tiny_e4b_host_weights(&config, 0x5eed);
    let mk = || {
        let guard = env_guard();
        let m = Gemma4E4bWgpu::new(
            Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
            &weights,
            64,
        )
        .unwrap();
        drop(guard);
        m
    };
    let n = 13usize;
    let mut stepped = Vec::new();
    {
        let mut m = mk();
        let mut t = m.decode_step(7).unwrap();
        stepped.push(t);
        for _ in 0..n {
            t = m.decode_step(t).unwrap();
            stepped.push(t);
        }
    }
    assert!(stepped.iter().all(|t| (*t as usize) < config.vocab_size));
    for k in [2usize, 3, 4, 8] {
        let mut m = mk();
        let mut t = m.decode_step(7).unwrap();
        let mut chained = vec![t];
        while chained.len() < n + 1 {
            let want = k.min(n + 1 - chained.len());
            let batch = m.decode_chain(t, want).unwrap();
            assert_eq!(batch.len(), want);
            t = *batch.last().unwrap();
            chained.extend(batch);
        }
        assert_eq!(
            stepped, chained,
            "chain k={k} diverged from stepped greedy decode"
        );
        eprintln!(
            "chain k={k}: {} tokens bit-identical to stepped",
            chained.len()
        );
    }
}

fn quantize_w4(l: &HostLin, gs: usize) -> HostLin {
    let (n, k) = (l.n, l.k);
    assert!(k % 32 == 0 && k % gs == 0);
    let mut packed = vec![0u32; n * k / 8];
    let mut scales = vec![0u16; n * (k / gs)];
    for r in 0..n {
        for g in 0..k / gs {
            let base = r * k + g * gs;
            let mut mx = 0f32;
            for i in 0..gs {
                mx = mx.max(half::bf16::from_bits(l.w[base + i]).to_f32().abs());
            }
            let sc = half::bf16::from_f32(if mx > 0.0 { mx / 7.0 } else { 1.0 });
            scales[r * (k / gs) + g] = sc.to_bits();
            let s = sc.to_f32();
            for i in 0..gs {
                let v = half::bf16::from_bits(l.w[base + i]).to_f32();
                let q = ((v / s).round() as i32 + 8).clamp(0, 15) as u32;
                let e = g * gs + i;
                packed[r * (k / 8) + e / 8] |= q << (4 * (e % 8));
            }
        }
    }
    HostLin::new_w4(packed, scales, gs, n, k)
}

#[test]
fn synthetic_w4_v4_routing_matches_block_bit_exact() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let mut weights = tiny_e4b_host_weights(&config, 0x5eed);
    for l in weights.layers.iter_mut() {
        l.qkv = quantize_w4(&l.qkv, 64);
        l.o = quantize_w4(&l.o, 32);
        l.gate_up = quantize_w4(&l.gate_up, 64);
        l.down = quantize_w4(&l.down, 32);
        l.per_layer_input_gate = quantize_w4(&l.per_layer_input_gate, 32);
    }
    let steps: Vec<u32> = vec![7, 300, 42, 511, 0, 13, 99, 250];

    let run = |label: &str, force_block: bool| -> Vec<(u32, Vec<f32>)> {
        let guard = env_guard();
        if force_block {
            std::env::set_var("NV_E4B_WGPU_W4_BLOCK", "1");
        } else {
            std::env::remove_var("NV_E4B_WGPU_W4_BLOCK");
        }
        std::env::set_var("NV_E4B_WGPU_W4_SG", "0");
        let mut m = Gemma4E4bWgpu::new(
            Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
            &weights,
            64,
        )
        .unwrap();
        std::env::remove_var("NV_E4B_WGPU_W4_BLOCK");
        std::env::remove_var("NV_E4B_WGPU_W4_SG");
        drop(guard);
        eprintln!("[{label}] passes per decode step: {}", m.pass_count());
        steps
            .iter()
            .map(|t| m.decode_step_logits(*t).unwrap())
            .collect()
    };

    let block = run("w4-block", true);
    let v4 = run("w4-v4", false);
    let mut diff_words = 0usize;
    for ((ta, la), (tb, lb)) in block.iter().zip(v4.iter()) {
        assert_eq!(ta, tb, "argmax token diverged between block and v4 routing");
        assert!(la.iter().all(|v| v.is_finite()));
        for (x, y) in la.iter().zip(lb.iter()) {
            if x.to_bits() != y.to_bits() {
                diff_words += 1;
            }
        }
    }
    assert_eq!(
        diff_words, 0,
        "w4 v4 routing must be bit-identical to the block path"
    );
    eprintln!(
        "w4 block-vs-v4: {} logits x {} steps bit-identical",
        config.vocab_size,
        steps.len()
    );
}

#[test]
fn synthetic_w4_sg16_routing_stays_close_to_v4() {
    let Some(ctx) = ctx_or_skip() else {
        return;
    };
    let width = ctx.subgroup_width();
    if !nv_kernels::wgpu_backend::kernels::gemv_w4a16::sg_pk_supported(width) {
        eprintln!("skipping: sg16 pk unsupported on subgroup width {width:?}");
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let mut weights = tiny_e4b_host_weights(&config, 0x5eed);
    for l in weights.layers.iter_mut() {
        l.qkv = quantize_w4(&l.qkv, 64);
        l.o = quantize_w4(&l.o, 32);
        l.gate_up = quantize_w4(&l.gate_up, 64);
        l.down = quantize_w4(&l.down, 32);
        l.per_layer_input_gate = quantize_w4(&l.per_layer_input_gate, 32);
    }
    let steps: Vec<u32> = vec![7, 300, 42, 511, 0, 13, 99, 250];

    let run = |label: &str, sg: bool| -> Vec<(u32, Vec<f32>)> {
        let guard = env_guard();
        std::env::set_var("NV_E4B_WGPU_W4_SG", if sg { "1" } else { "0" });
        let mut m = Gemma4E4bWgpu::new(
            Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
            &weights,
            64,
        )
        .unwrap();
        std::env::remove_var("NV_E4B_WGPU_W4_SG");
        drop(guard);
        eprintln!("[{label}] passes per decode step: {}", m.pass_count());
        steps
            .iter()
            .map(|t| m.decode_step_logits(*t).unwrap())
            .collect()
    };

    let v4 = run("w4-v4", false);
    let sg = run("w4-sg16", true);
    let mut max_abs = 0f32;
    for (i, ((ta, la), (tb, lb))) in v4.iter().zip(sg.iter()).enumerate() {
        assert!(
            lb.iter().all(|v| v.is_finite()),
            "step {i}: non-finite sg logits"
        );
        assert_eq!(
            ta, tb,
            "step {i}: argmax diverged between v4 and sg16 routing"
        );
        for (x, y) in la.iter().zip(lb.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
    }
    eprintln!(
        "w4 v4-vs-sg16: max abs logit diff {max_abs:.6e} over {} steps",
        steps.len()
    );
    assert!(
        max_abs < 0.25,
        "sg16 routing drifted from v4 by {max_abs} (different reduction order should stay within bf16 noise)"
    );
}

#[test]
fn synthetic_int8_lmhead_matches_bf16_head_on_exactly_representable_rows() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let mut weights = tiny_e4b_host_weights(&config, 0x1e4d);
    let s = 1.0f32 / 128.0;
    let mut rng = Lcg(0xabcd_1234);
    let hidden = config.hidden_size;
    for r in 0..config.vocab_size {
        for c in 0..hidden {
            let q = ((rng.next_f32() * 1270.0) as i32).clamp(-127, 127);
            let q = if c == 0 { 127 } else { q };
            weights.embed[r * hidden + c] = half::bf16::from_f32(q as f32 * s).to_bits();
        }
    }
    let steps: Vec<u32> = vec![7, 300, 42, 511, 0, 13, 99, 250];

    let run = |label: &str, int8: bool| -> Vec<(u32, Vec<f32>)> {
        let guard = env_guard();
        std::env::set_var("NV_E4B_LMHEAD_INT8", if int8 { "1" } else { "0" });
        let mut m = Gemma4E4bWgpu::new(
            Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
            &weights,
            64,
        )
        .unwrap();
        std::env::remove_var("NV_E4B_LMHEAD_INT8");
        drop(guard);
        eprintln!("[{label}] passes per decode step: {}", m.pass_count());
        steps
            .iter()
            .map(|t| m.decode_step_logits(*t).unwrap())
            .collect()
    };

    let bf = run("lmhead-bf16", false);
    let i8 = run("lmhead-int8", true);
    let mut max_abs = 0f32;
    let mut agree = 0usize;
    for (i, ((ta, la), (tb, lb))) in bf.iter().zip(i8.iter()).enumerate() {
        assert!(
            lb.iter().all(|v| v.is_finite()),
            "step {i}: non-finite int8 logits"
        );
        if ta == tb {
            agree += 1;
        }
        for (x, y) in la.iter().zip(lb.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
    }
    eprintln!(
        "int8 lm_head vs bf16 on exact rows: argmax {agree}/{} max abs diff {max_abs:.6e}",
        steps.len()
    );
    assert_eq!(
        agree,
        steps.len(),
        "argmax must agree on exactly-representable rows"
    );
    assert!(
        max_abs < 2e-2,
        "int8 lm_head drifted by {max_abs} on rows that quantize losslessly"
    );
}

#[test]
fn synthetic_sg_lmhead_is_bit_identical_to_tree_lmhead() {
    let Some(ctx) = ctx_or_skip() else { return };
    if !nv_kernels::wgpu_backend::kernels::gemv_bf16::sg32_ok(ctx) {
        eprintln!("skipping: needs a 32-wide subgroup adapter");
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let weights = tiny_e4b_host_weights(&config, 0x56c7);
    let steps: Vec<u32> = vec![7, 300, 42, 511, 0, 13, 99, 250];

    let run = |label: &str, setenv: &[(&str, &str)]| -> Vec<(u32, Vec<f32>)> {
        let guard = env_guard();
        for (k, v) in setenv {
            std::env::set_var(k, v);
        }
        let mut m = Gemma4E4bWgpu::new(
            Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
            &weights,
            64,
        )
        .unwrap();
        for (k, _) in setenv {
            std::env::remove_var(k);
        }
        drop(guard);
        eprintln!("[{label}] passes per decode step: {}", m.pass_count());
        steps
            .iter()
            .map(|t| m.decode_step_logits(*t).unwrap())
            .collect()
    };

    let tree = run("lmhead-tree", &[("NV_E4B_LMHEAD_SG", "0")]);
    for wg in ["128", "256"] {
        let sg = run(
            &format!("lmhead-sg-wg{wg}"),
            &[("NV_E4B_LMHEAD_SG", "1"), ("NV_E4B_LMHEAD_SG_WG", wg)],
        );
        for (i, ((ta, la), (tb, lb))) in tree.iter().zip(sg.iter()).enumerate() {
            assert_eq!(ta, tb, "step {i}: argmax diverged for sg lm_head wg{wg}");
            let diff = la
                .iter()
                .zip(lb.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                diff, 0,
                "step {i}: {diff} logit words differ bitwise for sg lm_head wg{wg}"
            );
        }
    }
}

type AttnRun = (
    Vec<(u32, Vec<f32>)>,
    Vec<Option<nv_models::gemma4_e4b_wgpu::KvCacheSnapshot>>,
);

#[test]
fn synthetic_fused_attn_matches_unfused_bitwise_including_kv_cache() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let n_layers = config.num_hidden_layers;
    let weights = tiny_e4b_host_weights(&config, 0xfac5);
    let steps: Vec<u32> = vec![7, 300, 42, 511, 0, 13, 99, 250, 17, 1];

    let run = |label: &str, fuse: &str| -> AttnRun {
        let guard = env_guard();
        std::env::set_var("NV_E4B_WGPU_FUSE_ATTN", fuse);
        let mut m = Gemma4E4bWgpu::new(
            Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
            &weights,
            64,
        )
        .unwrap();
        std::env::remove_var("NV_E4B_WGPU_FUSE_ATTN");
        drop(guard);
        eprintln!("[{label}] passes per decode step: {}", m.pass_count());
        let outs: Vec<(u32, Vec<f32>)> = steps
            .iter()
            .map(|t| m.decode_step_logits(*t).unwrap())
            .collect();
        let kvs: Vec<Option<nv_models::gemma4_e4b_wgpu::KvCacheSnapshot>> = (0..n_layers)
            .map(|li| m.kv_cache_snapshot(li).unwrap())
            .collect();
        (outs, kvs)
    };

    let (unfused, kv_unfused) = run("attn-unfused", "0");
    let (fused, kv_fused) = run("attn-fused", "1");
    let (default_env, kv_default) = {
        let guard = env_guard();
        std::env::remove_var("NV_E4B_WGPU_FUSE_ATTN");
        let mut m = Gemma4E4bWgpu::new(
            Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
            &weights,
            64,
        )
        .unwrap();
        drop(guard);
        let outs: Vec<(u32, Vec<f32>)> = steps
            .iter()
            .map(|t| m.decode_step_logits(*t).unwrap())
            .collect();
        let kvs: Vec<Option<nv_models::gemma4_e4b_wgpu::KvCacheSnapshot>> = (0..n_layers)
            .map(|li| m.kv_cache_snapshot(li).unwrap())
            .collect();
        (outs, kvs)
    };
    for (i, ((ta, la), (tb, lb))) in unfused.iter().zip(default_env.iter()).enumerate() {
        assert_eq!(ta, tb, "step {i}: default env must equal fusion-off");
        assert!(
            la.iter()
                .zip(lb.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "step {i}: default env logits must be bitwise equal to fusion-off (attn fusion must default OFF)"
        );
    }
    assert_eq!(
        kv_unfused.len(),
        kv_default.len(),
        "default env kv layer count"
    );

    for (i, ((ta, la), (tb, lb))) in unfused.iter().zip(fused.iter()).enumerate() {
        assert_eq!(ta, tb, "step {i}: argmax diverged fused vs unfused");
        let diff = la
            .iter()
            .zip(lb.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(diff, 0, "step {i}: {diff} logit words differ bitwise");
    }
    let mut compared = 0usize;
    for (li, (a, b)) in kv_unfused.iter().zip(kv_fused.iter()).enumerate() {
        match (a, b) {
            (None, None) => {}
            (Some((ka, va, ksa, vsa)), Some((kb, vb, ksb, vsb))) => {
                let dk = ka.iter().zip(kb.iter()).filter(|(x, y)| x != y).count();
                let dv = va.iter().zip(vb.iter()).filter(|(x, y)| x != y).count();
                let dks = ksa
                    .iter()
                    .zip(ksb.iter())
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count();
                let dvs = vsa
                    .iter()
                    .zip(vsb.iter())
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count();
                eprintln!("layer {li}: kv diff words k={dk} v={dv} k_scales={dks} v_scales={dvs}");
                assert_eq!(dk + dv, 0, "layer {li}: fp8 KV cache words differ");
                assert_eq!(dks + dvs, 0, "layer {li}: KV scales differ");
                compared += 1;
            }
            _ => panic!("layer {li}: kv cache presence differs fused vs unfused"),
        }
    }
    assert!(
        compared >= 2,
        "expected at least two owned-KV layers to compare, got {compared}"
    );
    let has_full = config
        .layer_types
        .iter()
        .any(|k| matches!(k, LayerType::FullAttention));
    assert!(has_full, "tiny config must exercise both head configs");
}

#[test]
fn synthetic_fused_head_argmax_matches_unfused_bitwise() {
    if ctx_or_skip().is_none() {
        return;
    }
    let cfg_capped = TINY_E4B_CONFIG.to_string();
    let cfg_uncapped = TINY_E4B_CONFIG.replace(
        "\"final_logit_softcapping\": 30.0",
        "\"final_logit_softcapping\": null",
    );
    assert_ne!(
        cfg_capped, cfg_uncapped,
        "config rewrite must hit the softcap field"
    );
    let steps: Vec<u32> = vec![7, 300, 42, 511, 0, 13, 99, 250];
    for (variant, cfg) in [("softcap", &cfg_capped), ("cast", &cfg_uncapped)] {
        let config = Gemma4Config::from_hf_json_str(cfg).unwrap();
        let weights = tiny_e4b_host_weights(&config, 0x4ead);
        let run = |label: &str, fuse: Option<&str>| -> (usize, Vec<(u32, Vec<f32>)>) {
            let guard = env_guard();
            match fuse {
                Some(v) => std::env::set_var("NV_E4B_WGPU_FUSE_HEAD", v),
                None => std::env::remove_var("NV_E4B_WGPU_FUSE_HEAD"),
            }
            let m = Gemma4E4bWgpu::new(Gemma4Config::from_hf_json_str(cfg).unwrap(), &weights, 64);
            std::env::remove_var("NV_E4B_WGPU_FUSE_HEAD");
            drop(guard);
            let mut m = m.unwrap();
            eprintln!(
                "[{variant}/{label}] passes per decode step: {}",
                m.pass_count()
            );
            (
                m.pass_count(),
                steps
                    .iter()
                    .map(|t| m.decode_step_logits(*t).unwrap())
                    .collect(),
            )
        };
        let (p_off, off) = run("head-unfused", Some("0"));
        let (p_def, def) = run("head-default", None);
        let (p_on, on) = run("head-fused", Some("1"));
        assert_eq!(
            p_def, p_off,
            "{variant}: default env must emit the unfused pass list (head fold defaults OFF)"
        );
        assert_eq!(
            p_on + 1,
            p_off,
            "{variant}: the fold must remove exactly one dispatch per step"
        );
        for (i, ((ta, la), (tb, lb))) in off.iter().zip(def.iter()).enumerate() {
            assert_eq!(ta, tb, "{variant} step {i}: default env token diverged");
            assert!(
                la.iter()
                    .zip(lb.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "{variant} step {i}: default env logits must equal fold-off bitwise"
            );
        }
        for (i, ((ta, la), (tb, lb))) in off.iter().zip(on.iter()).enumerate() {
            assert_eq!(
                ta, tb,
                "{variant} step {i}: argmax diverged fused vs unfused"
            );
            let diff = la
                .iter()
                .zip(lb.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                diff, 0,
                "{variant} step {i}: {diff} logit words differ bitwise"
            );
        }
    }
}

fn prefill_vs_stepped(
    label: &str,
    weights: &E4bHostWeights,
    seq: &[u32],
    setenv: &[(&str, &str)],
) -> (usize, usize) {
    let mk = || {
        let guard = env_guard();
        std::env::set_var("NV_E4B_WGPU_PREFILL_TAIL", "0");
        for (k, v) in setenv {
            std::env::set_var(k, v);
        }
        let m = Gemma4E4bWgpu::new(
            Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
            weights,
            64,
        )
        .unwrap();
        for (k, _) in setenv {
            std::env::remove_var(k);
        }
        std::env::remove_var("NV_E4B_WGPU_PREFILL_TAIL");
        drop(guard);
        m
    };

    let mut a = mk();
    let stepped: Vec<(u32, Vec<f32>)> = seq
        .iter()
        .map(|t| a.decode_step_logits(*t).unwrap())
        .collect();

    let mut b = mk();
    let m = b.prefill_chunk_len();
    assert!(m >= 2, "prefill disabled in build");
    let n_pre = b.prefill_tokens(seq).unwrap();
    assert!(n_pre >= m, "prefill_tokens consumed nothing");
    assert_eq!(n_pre % m, 0);
    assert_eq!(b.current_pos(), n_pre);
    let tail: Vec<(u32, Vec<f32>)> = seq[n_pre..]
        .iter()
        .map(|t| b.decode_step_logits(*t).unwrap())
        .collect();
    assert!(!tail.is_empty(), "seq must extend past the prefill chunks");

    let mut diff_words = 0usize;
    let mut max_abs = 0f32;
    for (i, ((ta, la), (tb, lb))) in stepped[n_pre..].iter().zip(tail.iter()).enumerate() {
        assert!(
            lb.iter().all(|v| v.is_finite()),
            "step {i}: non-finite logits"
        );
        assert_eq!(
            ta,
            tb,
            "[{label}] step {} argmax diverged after prefill",
            n_pre + i
        );
        for (x, y) in la.iter().zip(lb.iter()) {
            if x.to_bits() != y.to_bits() {
                diff_words += 1;
                max_abs = max_abs.max((x - y).abs());
            }
        }
    }
    eprintln!(
        "[{label}] prefill {} tok in {} chunks of {m}, {} stepped tail steps, {} differing logit words (max abs {max_abs:.3e})",
        n_pre,
        n_pre / m,
        tail.len(),
        diff_words
    );
    (n_pre, diff_words)
}

#[test]
fn synthetic_prefill_matches_stepped_decode_bitwise() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let weights = tiny_e4b_host_weights(&config, 0x5eed);
    let seq: Vec<u32> = vec![
        7, 300, 42, 511, 0, 13, 99, 250, 8, 9, 10, 11, 407, 3, 68, 133, 27, 501, 254, 77, 60, 61,
        62, 63, 64,
    ];
    let (n_pre, diffs) = prefill_vs_stepped("bf16", &weights, &seq, &[]);
    assert_eq!(n_pre, 20, "default chunk M must be 10");
    assert_eq!(
        diffs, 0,
        "prefill-then-decode logits must be bit-identical to stepped decode"
    );
}

#[test]
fn synthetic_prefill_w4_variants_match_stepped_bitwise() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let mut weights = tiny_e4b_host_weights(&config, 0x5eed);
    for l in weights.layers.iter_mut() {
        l.qkv = quantize_w4(&l.qkv, 64);
        l.o = quantize_w4(&l.o, 32);
        l.gate_up = quantize_w4(&l.gate_up, 64);
        l.down = quantize_w4(&l.down, 32);
        l.per_layer_input_gate = quantize_w4(&l.per_layer_input_gate, 32);
    }
    let seq: Vec<u32> = vec![
        7, 300, 42, 511, 0, 13, 99, 250, 8, 9, 10, 11, 407, 3, 68, 133, 27, 501, 254, 77, 60, 61,
        62, 63, 64,
    ];
    let (_, d_v4) = prefill_vs_stepped("w4-v4", &weights, &seq, &[("NV_E4B_WGPU_W4_SG", "0")]);
    assert_eq!(d_v4, 0, "w4 v4 prefill must be bit-identical to stepped");
    let (_, d_blk) = prefill_vs_stepped(
        "w4-block",
        &weights,
        &seq,
        &[("NV_E4B_WGPU_W4_SG", "0"), ("NV_E4B_WGPU_W4_BLOCK", "1")],
    );
    assert_eq!(
        d_blk, 0,
        "w4 block prefill must be bit-identical to stepped"
    );
    let (_, d_sg) = prefill_vs_stepped("w4-sg16", &weights, &seq, &[("NV_E4B_WGPU_W4_SG", "1")]);
    assert_eq!(
        d_sg, 0,
        "w4 sg16 prefill must be bit-identical to stepped now that mk routes through the sg kernels"
    );
}

#[test]
fn synthetic_prefill_m16_variants_match_stepped_bitwise() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let bf16_weights = tiny_e4b_host_weights(&config, 0x5eed);
    let mut w4_weights = tiny_e4b_host_weights(&config, 0x5eed);
    for l in w4_weights.layers.iter_mut() {
        l.qkv = quantize_w4(&l.qkv, 64);
        l.o = quantize_w4(&l.o, 32);
        l.gate_up = quantize_w4(&l.gate_up, 64);
        l.down = quantize_w4(&l.down, 32);
        l.per_layer_input_gate = quantize_w4(&l.per_layer_input_gate, 32);
    }
    let seq: Vec<u32> = (0..21u32).map(|i| (i * 37 + 5) % 512).collect();
    let m16 = ("NV_E4B_WGPU_PREFILL_M", "16");
    let (n_pre, d_bf16) = prefill_vs_stepped("bf16-m16", &bf16_weights, &seq, &[m16]);
    assert_eq!(n_pre, 16);
    assert_eq!(
        d_bf16, 0,
        "bf16 m16 prefill must be bit-identical to stepped"
    );
    let (_, d_v4) = prefill_vs_stepped(
        "w4-v4-m16",
        &w4_weights,
        &seq,
        &[m16, ("NV_E4B_WGPU_W4_SG", "0")],
    );
    assert_eq!(
        d_v4, 0,
        "w4 v4 m16 prefill must be bit-identical to stepped"
    );
    let (_, d_blk) = prefill_vs_stepped(
        "w4-block-m16",
        &w4_weights,
        &seq,
        &[
            m16,
            ("NV_E4B_WGPU_W4_SG", "0"),
            ("NV_E4B_WGPU_W4_BLOCK", "1"),
        ],
    );
    assert_eq!(
        d_blk, 0,
        "w4 block m16 prefill must be bit-identical to stepped"
    );
    let (_, d_sg) = prefill_vs_stepped(
        "w4-sg16-m16",
        &w4_weights,
        &seq,
        &[m16, ("NV_E4B_WGPU_W4_SG", "1")],
    );
    assert_eq!(
        d_sg, 0,
        "w4 sg16 m16 prefill must be bit-identical to stepped"
    );
}

fn prefill_tail_vs_stepped(
    label: &str,
    weights: &E4bHostWeights,
    seq: &[u32],
    pre_len: usize,
    setenv: &[(&str, &str)],
) {
    let mk = || {
        let guard = env_guard();
        for (k, v) in setenv {
            std::env::set_var(k, v);
        }
        let m = Gemma4E4bWgpu::new(
            Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
            weights,
            64,
        )
        .unwrap();
        for (k, _) in setenv {
            std::env::remove_var(k);
        }
        drop(guard);
        m
    };
    let mut a = mk();
    let stepped: Vec<(u32, Vec<f32>)> = seq
        .iter()
        .map(|t| a.decode_step_logits(*t).unwrap())
        .collect();
    let mut b = mk();
    let m = b.prefill_chunk_len();
    assert!(
        m >= 2 && !pre_len.is_multiple_of(m),
        "pre_len must leave a partial tail"
    );
    let done = b.prefill_tokens(&seq[..pre_len]).unwrap();
    assert_eq!(done, pre_len, "tail chunk must consume the whole remainder");
    assert_eq!(b.current_pos(), pre_len);
    let mut diff_words = 0usize;
    for (i, t) in seq[pre_len..].iter().enumerate() {
        let (tb, lb) = b.decode_step_logits(*t).unwrap();
        let (ta, la) = &stepped[pre_len + i];
        assert_eq!(
            *ta,
            tb,
            "[{label}] step {} argmax diverged after tail prefill",
            pre_len + i
        );
        diff_words += la
            .iter()
            .zip(lb.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
    }
    eprintln!(
        "[{label}] tail prefill {pre_len} tok (chunk {m}), {} stepped tail steps, {diff_words} differing logit words",
        seq.len() - pre_len
    );
    assert_eq!(
        diff_words, 0,
        "[{label}] tail-chunk prefill must be bit-identical to stepped"
    );
}

#[test]
fn synthetic_prefill_tail_chunk_matches_stepped_bitwise() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let bf16_weights = tiny_e4b_host_weights(&config, 0x5eed);
    let mut w4_weights = tiny_e4b_host_weights(&config, 0x5eed);
    for l in w4_weights.layers.iter_mut() {
        l.qkv = quantize_w4(&l.qkv, 64);
        l.o = quantize_w4(&l.o, 32);
        l.gate_up = quantize_w4(&l.gate_up, 64);
        l.down = quantize_w4(&l.down, 32);
        l.per_layer_input_gate = quantize_w4(&l.per_layer_input_gate, 32);
    }
    let seq: Vec<u32> = (0..24u32).map(|i| (i * 53 + 11) % 512).collect();
    prefill_tail_vs_stepped("bf16-tail", &bf16_weights, &seq, 21, &[]);
    prefill_tail_vs_stepped(
        "w4-sg16-tail",
        &w4_weights,
        &seq,
        21,
        &[("NV_E4B_WGPU_W4_SG", "1")],
    );
    prefill_tail_vs_stepped(
        "w4-v4-tail",
        &w4_weights,
        &seq,
        21,
        &[("NV_E4B_WGPU_W4_SG", "0")],
    );
    prefill_tail_vs_stepped("bf16-tail-short", &bf16_weights, &seq, 5, &[]);
}

fn bits_to_tensor(bits: &[u16], shape: &[usize], dev: &candle_core::Device) -> candle_core::Tensor {
    let v: Vec<half::bf16> = bits.iter().map(|b| half::bf16::from_bits(*b)).collect();
    candle_core::Tensor::from_vec(v, shape, dev)
        .unwrap()
        .to_dtype(candle_core::DType::BF16)
        .unwrap()
}

fn write_safetensors(config: &Gemma4Config, w: &E4bHostWeights, path: &std::path::Path) {
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;
    let dev = Device::Cpu;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let hpl = config.hidden_size_per_layer_input;
    let n_layers = config.num_hidden_layers;
    let ple_row = n_layers * hpl;
    let p0 = "model.language_model";

    map.insert(
        format!("{p0}.embed_tokens.weight"),
        bits_to_tensor(&w.embed, &[config.vocab_size, hidden], &dev),
    );
    map.insert(
        format!("{p0}.embed_tokens_per_layer.weight"),
        bits_to_tensor(
            &w.embed_per_layer,
            &[config.vocab_size_per_layer(), ple_row],
            &dev,
        ),
    );
    map.insert(
        format!("{p0}.per_layer_model_projection.weight"),
        bits_to_tensor(&w.per_layer_model_projection.w, &[ple_row, hidden], &dev),
    );
    map.insert(
        format!("{p0}.per_layer_projection_norm.weight"),
        bits_to_tensor(&w.per_layer_projection_norm, &[hpl], &dev),
    );
    map.insert(
        format!("{p0}.norm.weight"),
        bits_to_tensor(&w.final_norm, &[hidden], &dev),
    );

    for (i, l) in w.layers.iter().enumerate() {
        let p = format!("{p0}.layers.{i}");
        let kind = l.kind;
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let n_q = config.num_attention_heads;
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        map.insert(
            format!("{p}.self_attn.q_proj.weight"),
            bits_to_tensor(&l.qkv.w[..q_dim * hidden], &[q_dim, hidden], &dev),
        );
        if l.kv_source.is_none() {
            map.insert(
                format!("{p}.self_attn.k_proj.weight"),
                bits_to_tensor(
                    &l.qkv.w[q_dim * hidden..(q_dim + kv_dim) * hidden],
                    &[kv_dim, hidden],
                    &dev,
                ),
            );
            map.insert(
                format!("{p}.self_attn.v_proj.weight"),
                bits_to_tensor(
                    &l.qkv.w[(q_dim + kv_dim) * hidden..],
                    &[kv_dim, hidden],
                    &dev,
                ),
            );
            map.insert(
                format!("{p}.self_attn.k_norm.weight"),
                bits_to_tensor(&l.k_norm, &[hd], &dev),
            );
        }
        map.insert(
            format!("{p}.self_attn.o_proj.weight"),
            bits_to_tensor(&l.o.w, &[hidden, q_dim], &dev),
        );
        map.insert(
            format!("{p}.self_attn.q_norm.weight"),
            bits_to_tensor(&l.q_norm, &[hd], &dev),
        );
        map.insert(
            format!("{p}.mlp.gate_proj.weight"),
            bits_to_tensor(&l.gate_up.w[..inter * hidden], &[inter, hidden], &dev),
        );
        map.insert(
            format!("{p}.mlp.up_proj.weight"),
            bits_to_tensor(&l.gate_up.w[inter * hidden..], &[inter, hidden], &dev),
        );
        map.insert(
            format!("{p}.mlp.down_proj.weight"),
            bits_to_tensor(&l.down.w, &[hidden, inter], &dev),
        );
        map.insert(
            format!("{p}.per_layer_input_gate.weight"),
            bits_to_tensor(&l.per_layer_input_gate.w, &[hpl, hidden], &dev),
        );
        map.insert(
            format!("{p}.per_layer_projection.weight"),
            bits_to_tensor(&l.per_layer_projection.w, &[hidden, hpl], &dev),
        );
        for (suffix, bits) in [
            ("input_layernorm", &l.input_ln),
            ("post_attention_layernorm", &l.post_attn_ln),
            ("pre_feedforward_layernorm", &l.pre_ff_ln),
            ("post_feedforward_layernorm", &l.post_ff_ln),
            ("post_per_layer_input_norm", &l.post_per_layer_input_norm),
        ] {
            map.insert(
                format!("{p}.{suffix}.weight"),
                bits_to_tensor(bits, &[hidden], &dev),
            );
        }
        map.insert(
            format!("{p}.layer_scalar"),
            Tensor::from_vec(vec![half::bf16::from_f32(l.layer_scalar)], &[1], &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        );
    }
    candle_core::safetensors::save(&map, path).unwrap();
}

#[test]
fn synthetic_from_loader_matches_host_build() {
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
    let mut weights = tiny_e4b_host_weights(&config, 0x10ade4);
    for l in &mut weights.layers {
        l.layer_scalar = half::bf16::from_f32(l.layer_scalar).to_f32();
    }
    let dir = std::path::PathBuf::from(
        std::env::var("NV_E4B_WGPU_TMP")
            .unwrap_or_else(|_| format!("{}/.cache/nvk-tmp", std::env::var("HOME").unwrap())),
    )
    .join(format!("e4bwgpu-stream-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let st = dir.join("model.safetensors");
    write_safetensors(&config, &weights, &st);
    let loader = nv_weights::WeightLoader::open_file(&st, &candle_core::Device::Cpu).unwrap();

    let steps: Vec<u32> = vec![7, 300, 42, 511, 0, 13, 99, 250, 8, 9, 10, 11];
    let guard = env_guard();
    let mut host_m = Gemma4E4bWgpu::new(config.clone(), &weights, 64).unwrap();
    let mut loader_m = Gemma4E4bWgpu::from_loader(config.clone(), &loader, 64).unwrap();
    drop(guard);
    assert_eq!(host_m.pass_count(), loader_m.pass_count());
    assert_eq!(
        host_m.weight_bytes_per_token(),
        loader_m.weight_bytes_per_token()
    );
    for (si, t) in steps.iter().enumerate() {
        let (ta, la) = host_m.decode_step_logits(*t).unwrap();
        let (tb, lb) = loader_m.decode_step_logits(*t).unwrap();
        assert_eq!(ta, tb, "step {si}: argmax diverged");
        let mismatch = la
            .iter()
            .zip(lb.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(mismatch, 0, "step {si}: {mismatch} logits differ bitwise");
    }
    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "from_loader vs host build: {} steps x {} logits bit-identical",
        steps.len(),
        config.vocab_size
    );
}

#[cfg(feature = "cuda")]
mod cuda_compare {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn softcap(v: f32, cap: f32) -> f32 {
        if cap > 0.0 && cap.is_finite() {
            cap * (v / cap).tanh()
        } else {
            v
        }
    }

    fn compare_against_cuda(label: &str, weights: E4bHostWeights, rel_bound: f32) {
        if ctx_or_skip().is_none() {
            return;
        }
        let cuda_dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                if require() {
                    panic!("no CUDA device: {e}");
                }
                eprintln!("skipping: no CUDA device ({e})");
                return;
            }
        };
        let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();

        let dir =
            std::path::PathBuf::from(std::env::var("NV_E4B_WGPU_TMP").unwrap_or_else(|_| {
                format!("{}/.cache/nvk-tmp", std::env::var("HOME").unwrap())
            }))
            .join(format!("e4bwgpu-cmp-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let st = dir.join("model.safetensors");
        write_safetensors(&config, &weights, &st);

        let loader = nv_weights::WeightLoader::open_file(&st, &cuda_dev).unwrap();
        let model = nv_models::gemma4_e4b::Gemma4E4b::from_loader(
            Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
            &loader,
            &cuda_dev,
        )
        .unwrap();

        let mut wg = Gemma4E4bWgpu::new(
            Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap(),
            &weights,
            64,
        )
        .unwrap();
        eprintln!("wgpu passes per decode step: {}", wg.pass_count());

        let cap = config.final_logit_softcapping;
        let mut cache: Vec<Option<(Tensor, Tensor)>> = vec![None; config.num_hidden_layers];
        let steps: Vec<u32> = vec![7, 300, 42, 511, 0, 13, 99, 250, 8, 9, 10, 11];
        let mut worst_abs = 0f32;
        let mut worst_rel = 0f32;
        let mut argmax_match = 0usize;
        let mut worst_regret = 0f32;
        for (si, tok) in steps.iter().enumerate() {
            let past = wg.current_pos();
            let (logits, _) = model.forward_step_spec(&[*tok], past, &mut cache).unwrap();
            let cuda_logits: Vec<f32> = logits
                .flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(|v| softcap(v, cap))
                .collect();
            let (wg_tok, wg_logits) = wg.decode_step_logits(*tok).unwrap();

            let cuda_argmax = cuda_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            let mut max_abs = 0f32;
            let mut max_rel = 0f32;
            for (c, w) in cuda_logits.iter().zip(wg_logits.iter()) {
                let d = (c - w).abs();
                if d > max_abs {
                    max_abs = d;
                }
                let r = d / c.abs().max(1e-6);
                if r > max_rel {
                    max_rel = r;
                }
            }
            worst_abs = worst_abs.max(max_abs);
            worst_rel = worst_rel.max(max_rel);
            if cuda_argmax == wg_tok {
                argmax_match += 1;
            }
            let cuda_top = cuda_logits[cuda_argmax as usize];
            let cuda_at_wg = cuda_logits[wg_tok as usize];
            let regret = cuda_top - cuda_at_wg;
            worst_regret = worst_regret.max(regret);
            let mut sorted = cuda_logits.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let gap = sorted[0] - sorted[1];
            eprintln!(
                "step {si}: tok {tok} cuda_argmax {cuda_argmax} wgpu_argmax {wg_tok} max_abs {max_abs:.6e} max_rel {max_rel:.6e} cuda_top1_top2_gap {gap:.6e} regret {regret:.6e}"
            );
            assert!(wg_logits.iter().all(|v| v.is_finite()));
            assert!(cuda_logits.iter().all(|v| v.is_finite()));
        }
        eprintln!(
            "[{label}] cuda-vs-wgpu E4B over {} steps: worst max_abs {worst_abs:.6e} worst max_rel {worst_rel:.6e} worst argmax regret {worst_regret:.6e} exact argmax agreement {argmax_match}/{}",
            steps.len(),
            steps.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            worst_rel < rel_bound,
            "[{label}] cuda-vs-wgpu E4B logits diverged: worst max_rel {worst_rel} >= {rel_bound}"
        );
        assert!(
            worst_regret <= 2.0 * worst_abs,
            "[{label}] wgpu picked a token CUDA ranks {worst_regret} below its own top-1, which is more than 2x the observed logit error {worst_abs}"
        );
    }

    #[test]
    fn cuda_vs_wgpu_e4b_synthetic_decode_logits() {
        let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
        compare_against_cuda("baseline", tiny_e4b_host_weights(&config, 0xc0de), 0.02);
    }

    #[test]
    fn cuda_vs_wgpu_e4b_per_layer_path_amplified() {
        let config = Gemma4Config::from_hf_json_str(TINY_E4B_CONFIG).unwrap();
        let mut w = tiny_e4b_host_weights(&config, 0xa11ce);
        let amp = |v: &mut Vec<u16>, s: f32| {
            for x in v.iter_mut() {
                *x = half::bf16::from_f32(half::bf16::from_bits(*x).to_f32() * s).to_bits();
            }
        };
        for l in w.layers.iter_mut() {
            amp(&mut l.per_layer_projection.w, 8.0);
            amp(&mut l.per_layer_input_gate.w, 4.0);
        }
        amp(&mut w.embed_per_layer, 4.0);
        amp(&mut w.per_layer_model_projection.w, 4.0);
        compare_against_cuda("per-layer-amplified", w, 0.02);
    }
}

fn free_vram_mib() -> Option<u64> {
    let o = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&o.stdout).trim().parse().ok()
}

fn truncate_layers(w: &mut E4bHostWeights, full_layers: usize, keep: usize, hpl: usize) {
    if keep >= full_layers {
        return;
    }
    w.layers.truncate(keep);
    let full_row = full_layers * hpl;
    let new_row = keep * hpl;
    let rows = w.embed_per_layer.len() / full_row;
    let mut out = Vec::with_capacity(rows * new_row);
    for r in 0..rows {
        out.extend_from_slice(&w.embed_per_layer[r * full_row..r * full_row + new_row]);
    }
    w.embed_per_layer = out;
    let k = w.per_layer_model_projection.k;
    if let Some(q) = &mut w.per_layer_model_projection.q {
        q.packed.truncate(new_row * k / 8);
        q.scales.truncate(new_row * (k / q.gs));
    } else {
        w.per_layer_model_projection.w.truncate(new_row * k);
    }
    w.per_layer_model_projection.n = new_row;
}

#[test]
#[ignore]
fn real_e4b_checkpoint_wgpu_decode() {
    if std::env::var("NV_E4B_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_E4B_WGPU_TEST=1 to run");
        return;
    }
    let Some(_ctx) = ctx_or_skip() else { return };
    let dir = e4b_snapshot_dir();
    eprintln!("E4B checkpoint: {}", dir.display());
    eprintln!("free VRAM before load: {:?} MiB", free_vram_mib());

    let mut config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let full_layers = config.num_hidden_layers;
    assert!(
        config.has_per_layer_embeddings(),
        "E4B must carry per-layer embeddings"
    );
    assert!(
        config.num_kv_shared_layers > 0,
        "E4B must carry KV-shared layers"
    );
    let keep: usize = std::env::var("NV_E4B_WGPU_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(full_layers)
        .min(full_layers);

    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let quantized = loader.has("model.language_model.layers.0.mlp.down_proj.weight_packed");
    eprintln!(
        "checkpoint weight storage: {}",
        if quantized {
            "w4a16 pack-quantized"
        } else {
            "bf16"
        }
    );
    for i in 0..keep.min(6) {
        let p = format!("model.language_model.layers.{i}");
        if quantized {
            assert!(
                loader.has(&format!("{p}.mlp.down_proj.weight_packed"))
                    && loader.has(&format!("{p}.mlp.down_proj.weight_scale")),
                "layer {i} down_proj must carry packed weights + scales"
            );
            assert!(
                loader.has(&format!("{p}.per_layer_input_gate.weight_packed"))
                    && loader.has(&format!("{p}.per_layer_projection.weight_packed")),
                "layer {i} must carry the per-layer embedding projections"
            );
        } else {
            assert_eq!(
                loader.st_dtype_of(&format!("{p}.mlp.down_proj.weight")),
                Some(nv_weights::StDtype::BF16),
                "layer {i} down_proj must be stored BF16"
            );
            assert!(
                loader.has(&format!("{p}.per_layer_input_gate.weight"))
                    && loader.has(&format!("{p}.per_layer_projection.weight")),
                "layer {i} must carry the per-layer embedding projections"
            );
        }
    }

    let max_seq: usize = std::env::var("NV_E4B_WGPU_MAXSEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let stream = std::env::var("NV_E4B_WGPU_STREAM").ok().as_deref() == Some("1");
    let t_build = std::time::Instant::now();
    let mut m = if stream {
        assert_eq!(
            keep, full_layers,
            "NV_E4B_WGPU_STREAM=1 streams the full checkpoint; unset NV_E4B_WGPU_LAYERS"
        );
        let m = Gemma4E4bWgpu::from_loader(config.clone(), &loader, max_seq).unwrap();
        eprintln!(
            "Gemma4E4bWgpu::from_loader: {:.1}s, {} passes/token, {:.2} GiB weights/token",
            t_build.elapsed().as_secs_f64(),
            m.pass_count(),
            m.weight_bytes_per_token() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        m
    } else {
        let t_load = std::time::Instant::now();
        let mut host = e4b_host_weights_from_loader(&config, &loader).unwrap();
        eprintln!(
            "e4b_host_weights_from_loader: OK, {:.1}s",
            t_load.elapsed().as_secs_f64()
        );
        if keep < full_layers {
            eprintln!("truncating {full_layers} -> {keep} layers to bound VRAM");
            truncate_layers(
                &mut host,
                full_layers,
                keep,
                config.hidden_size_per_layer_input,
            );
            config.num_hidden_layers = keep;
            config.layer_types.truncate(keep);
            config.num_kv_shared_layers = config.num_kv_shared_layers.min(keep / 2);
            for (i, l) in host.layers.iter_mut().enumerate() {
                l.kv_source = config.kv_source_layer(i);
            }
        }
        assert_eq!(host.layers.len(), config.num_hidden_layers);
        let t_new = std::time::Instant::now();
        let m = Gemma4E4bWgpu::new(config.clone(), &host, max_seq).unwrap();
        eprintln!(
            "Gemma4E4bWgpu::new: {:.1}s, {} passes/token, {:.2} GiB weights/token",
            t_new.elapsed().as_secs_f64(),
            m.pass_count(),
            m.weight_bytes_per_token() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        m
    };
    eprintln!("free VRAM after upload: {:?} MiB", free_vram_mib());

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let id_of = |s: &str| tok.token_to_id(s);
    let bos = id_of("<bos>").expect("<bos>");
    let turn_open = id_of("<|turn>").expect("<|turn>");
    let turn_close = id_of("<turn|>").expect("<turn|>");
    let nl = id_of("\n").expect("newline token");
    let role_user = id_of("user").expect("user");
    let role_model = id_of("model").expect("model");
    let enc_text = |s: &str| -> Vec<u32> { tok.encode(s, false).unwrap().get_ids().to_vec() };

    let n_new: usize = std::env::var("NV_E4B_WGPU_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let eos: Vec<u32> = vec![1, turn_close, 50];

    let mut run = |label: &str, ids: Vec<u32>| -> (Vec<u32>, String, f64) {
        m.reset();
        assert!(!ids.is_empty());
        let t0 = std::time::Instant::now();
        let mut next = 0u32;
        for t in &ids {
            next = m.decode_step(*t).unwrap();
        }
        let prefill = t0.elapsed().as_secs_f64();
        let mut out = Vec::new();
        let chain_k = nv_models::gemma4_e4b_wgpu::chain_k_from_env();
        let mut pending: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        let t1 = std::time::Instant::now();
        for _ in 0..n_new {
            out.push(next);
            if eos.contains(&next) {
                break;
            }
            if chain_k > 1 {
                if pending.is_empty() {
                    let pos0 = m.current_pos();
                    let want = chain_k
                        .min(m.max_seq().saturating_sub(pos0))
                        .min(n_new.saturating_sub(out.len()))
                        .max(1);
                    let mut batch = m.decode_chain(next, want).unwrap();
                    if let Some(j) = batch.iter().position(|t| eos.contains(t)) {
                        if j + 1 < batch.len() {
                            batch.truncate(j + 1);
                            m.truncate_to(pos0 + j + 1).unwrap();
                        }
                    }
                    pending.extend(batch);
                }
                next = pending.pop_front().unwrap();
            } else {
                next = m.decode_step(next).unwrap();
            }
        }
        let dt = t1.elapsed().as_secs_f64();
        let ms = 1000.0 * dt / out.len().max(1) as f64;
        eprintln!(
            "[{label}] prompt {} tok, prefill {prefill:.3}s; decode {} tok in {dt:.3}s = {ms:.2} ms/tok",
            ids.len(),
            out.len()
        );
        eprintln!("[{label}] generated ids: {out:?}");
        let text = tok.decode(&out, false).unwrap();
        eprintln!("[{label}] generated text: {text:?}");
        (out, text, ms)
    };

    let mut chat = vec![bos, turn_open, role_user, nl];
    chat.extend(enc_text(
        &std::env::var("NV_E4B_WGPU_PROMPT")
            .unwrap_or_else(|_| "Name three primary colors.".to_string()),
    ));
    chat.extend([turn_close, nl, turn_open, role_model, nl]);
    let (chat_ids, chat_text, _) = run("chat", chat);

    let mut cont = vec![bos];
    cont.extend(enc_text("The capital of France is"));
    let (cont_ids, cont_text, cont_ms) = run("continuation", cont);

    eprintln!("free VRAM after decode: {:?} MiB", free_vram_mib());
    eprintln!(
        "E4B on wgpu: {} layers, {} passes/token, {:.2} ms/tok (co-tenanted GPU, not a clean benchmark)",
        config.num_hidden_layers,
        m.pass_count(),
        cont_ms
    );

    for (label, out) in [("chat", &chat_ids), ("continuation", &cont_ids)] {
        assert!(!out.is_empty(), "{label}: no tokens");
        assert!(
            out.iter().all(|t| (*t as usize) < config.vocab_size),
            "{label}: token out of vocab"
        );
    }
    if keep == full_layers {
        assert!(
            !chat_text.trim().is_empty() && !cont_text.trim().is_empty(),
            "full-depth E4B decode produced empty text"
        );
        let body: Vec<u32> = cont_ids
            .iter()
            .copied()
            .filter(|t| !eos.contains(t))
            .collect();
        let distinct: std::collections::HashSet<u32> = body.iter().copied().collect();
        assert!(
            distinct.len() * 3 >= body.len(),
            "continuation decode degenerated into a short repeating cycle: {distinct:?} distinct of {} tokens",
            body.len()
        );
        assert!(
            cont_text.to_lowercase().contains("paris"),
            "greedy continuation of \"The capital of France is\" should name Paris, got {cont_text:?}"
        );
    }
}

#[test]
#[ignore]
fn real_e4b_prefill_parity_and_ttft() {
    if std::env::var("NV_E4B_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_E4B_WGPU_TEST=1 to run");
        return;
    }
    let Some(_ctx) = ctx_or_skip() else { return };
    let dir = e4b_snapshot_dir();
    eprintln!("E4B checkpoint: {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let t_load = std::time::Instant::now();
    let host = e4b_host_weights_from_loader(&config, &loader).unwrap();
    eprintln!(
        "host weights loaded in {:.1}s",
        t_load.elapsed().as_secs_f64()
    );

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let bos = tok.token_to_id("<bos>").expect("<bos>");
    let sentence = "The history of navigation at sea spans thousands of years, from Polynesian wayfinding by stars and swells to the magnetic compass, the marine chronometer, and satellite positioning. ";
    let mk_prompt = |target: usize| -> Vec<u32> {
        let mut ids = vec![bos];
        while ids.len() < target {
            ids.extend(tok.encode(sentence, false).unwrap().get_ids());
        }
        ids.truncate(target);
        ids
    };
    let short: Vec<u32> = {
        let mut v = vec![bos];
        v.extend(
            tok.encode("The capital of France is", false)
                .unwrap()
                .get_ids(),
        );
        v
    };
    let prompts: Vec<(String, Vec<u32>)> = match std::env::var("NV_E4B_PF_PROMPTS") {
        Ok(s) => s
            .split(',')
            .filter_map(|p| p.trim().parse::<usize>().ok())
            .map(|len| (format!("len-{len}"), mk_prompt(len)))
            .collect(),
        Err(_) => vec![
            ("short".to_string(), short),
            ("mid-500".to_string(), mk_prompt(500)),
            ("long-2000".to_string(), mk_prompt(2000)),
        ],
    };

    let n_new: usize = std::env::var("NV_E4B_PF_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let max_seq = (2000 + n_new + 64).next_multiple_of(64);
    let t_new = std::time::Instant::now();
    let mut m = Gemma4E4bWgpu::new(config.clone(), &host, max_seq).unwrap();
    eprintln!(
        "Gemma4E4bWgpu::new: {:.1}s, {} decode passes, {} prefill passes, chunk {}",
        t_new.elapsed().as_secs_f64(),
        m.pass_count(),
        m.prefill_pass_count(),
        m.prefill_chunk_len()
    );
    assert!(m.prefill_chunk_len() >= 2, "prefill must be enabled");
    m.decode_step(5).unwrap();
    m.reset();
    m.prefill_chunk(&vec![5u32; m.prefill_chunk_len()]).unwrap();
    m.sync().unwrap();
    m.reset();

    for (label, prompt) in &prompts {
        m.reset();
        let t0 = std::time::Instant::now();
        let mut next = 0u32;
        for t in prompt {
            next = m.decode_step(*t).unwrap();
        }
        let ttft_stepped = t0.elapsed().as_secs_f64();
        let mut base = Vec::new();
        for _ in 0..n_new {
            base.push(next);
            next = m.decode_step(next).unwrap();
        }

        m.reset();
        let t1 = std::time::Instant::now();
        let mut next2 = m.prefill_prompt(prompt).unwrap();
        let ttft_pf = t1.elapsed().as_secs_f64();
        let mut gen = Vec::new();
        for _ in 0..n_new {
            gen.push(next2);
            next2 = m.decode_step(next2).unwrap();
        }

        eprintln!(
            "[{label}] prompt {} tok: TTFT stepped {:.3}s ({:.2} ms/tok) -> prefill {:.3}s ({:.2} ms/tok), speedup {:.2}x",
            prompt.len(),
            ttft_stepped,
            1000.0 * ttft_stepped / prompt.len() as f64,
            ttft_pf,
            1000.0 * ttft_pf / prompt.len() as f64,
            ttft_stepped / ttft_pf
        );
        eprintln!("[{label}] stepped ids: {base:?}\n[{label}] prefill ids: {gen:?}");
        let text = tok.decode(&gen, false).unwrap();
        eprintln!("[{label}] prefill text: {text:?}");
        assert_eq!(
            gen, base,
            "[{label}] greedy continuation diverged between prefill and stepped prefill"
        );
    }
}

#[test]
#[ignore]
fn real_e4b_prefill_chunk_profile() {
    if std::env::var("NV_E4B_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_E4B_WGPU_TEST=1 to run");
        return;
    }
    let Some(_ctx) = ctx_or_skip() else { return };
    let dir = e4b_snapshot_dir();
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = e4b_host_weights_from_loader(&config, &loader).unwrap();
    let max_seq = 2304;
    let mut m = Gemma4E4bWgpu::new(config.clone(), &host, max_seq).unwrap();
    let cm = m.prefill_chunk_len();
    assert!(cm >= 2);
    let ids: Vec<u32> = (0..cm as u32).map(|i| 1000 + i).collect();

    m.reset();
    m.prefill_chunk(&ids).unwrap();
    m.sync().unwrap();
    m.reset();
    let chunks_per_block = 256 / cm;
    for block in 0..8 {
        let t0 = std::time::Instant::now();
        for _ in 0..chunks_per_block {
            m.prefill_chunk(&ids).unwrap();
        }
        m.sync().unwrap();
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "[profile] ctx {:>4}..{:>4}: {:.1} ms/chunk, {:.2} ms/tok",
            block * 256,
            (block + 1) * 256,
            1000.0 * dt / chunks_per_block as f64,
            1000.0 * dt / 256.0
        );
    }
    let t0 = std::time::Instant::now();
    let n_dec = 64;
    let mut next = 5u32;
    for _ in 0..n_dec {
        next = m.decode_step(next).unwrap();
    }
    eprintln!(
        "[profile] decode at ctx 2048: {:.2} ms/tok",
        1000.0 * t0.elapsed().as_secs_f64() / n_dec as f64
    );
}

#[test]
#[ignore]
fn real_e4b_lmhead_int8_argmax_agreement() {
    if std::env::var("NV_E4B_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_E4B_WGPU_TEST=1 to run");
        return;
    }
    let Some(_ctx) = ctx_or_skip() else { return };
    let dir = e4b_snapshot_dir();
    eprintln!("E4B checkpoint: {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let t_load = std::time::Instant::now();
    let host = e4b_host_weights_from_loader(&config, &loader).unwrap();
    eprintln!(
        "host weights loaded in {:.1}s",
        t_load.elapsed().as_secs_f64()
    );

    let n_new: usize = std::env::var("NV_E4B_INT8_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(220)
        .max(200);
    let max_seq = 64 + n_new.next_multiple_of(64);

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let bos = tok.token_to_id("<bos>").expect("<bos>");
    let mut prompt = vec![bos];
    prompt.extend(
        tok.encode(
            "Write a detailed paragraph about the history of the compass.",
            false,
        )
        .unwrap()
        .get_ids()
        .to_vec(),
    );

    let build = |int8: bool| -> Gemma4E4bWgpu {
        let guard = env_guard();
        std::env::set_var("NV_E4B_LMHEAD_INT8", if int8 { "1" } else { "0" });
        let m = Gemma4E4bWgpu::new(config.clone(), &host, max_seq).unwrap();
        std::env::remove_var("NV_E4B_LMHEAD_INT8");
        drop(guard);
        m
    };

    let mut fed: Vec<u32> = prompt.clone();
    let mut bf16_argmax: Vec<u32> = Vec::new();
    {
        let mut a = build(false);
        let mut next = 0u32;
        for t in &prompt {
            next = a.decode_step(*t).unwrap();
        }
        for _ in 0..n_new {
            bf16_argmax.push(next);
            fed.push(next);
            next = a.decode_step(next).unwrap();
        }
        fed.pop();
    }

    let mut b = build(true);
    let mut i8_argmax: Vec<u32> = Vec::new();
    for (i, t) in fed.iter().enumerate() {
        let out = b.decode_step(*t).unwrap();
        if i + 1 >= prompt.len() {
            i8_argmax.push(out);
        }
    }
    assert_eq!(i8_argmax.len(), bf16_argmax.len());

    let agree = bf16_argmax
        .iter()
        .zip(i8_argmax.iter())
        .filter(|(a, b)| a == b)
        .count();
    let rate = agree as f64 / bf16_argmax.len() as f64;
    let mut first_diff = None;
    for (i, (a, b)) in bf16_argmax.iter().zip(i8_argmax.iter()).enumerate() {
        if a != b {
            first_diff = Some((i, *a, *b));
            break;
        }
    }
    eprintln!(
        "int8 lm_head argmax agreement (teacher-forced on the bf16 greedy stream): {agree}/{} = {rate:.4}, first divergence {first_diff:?}",
        bf16_argmax.len()
    );
    assert!(
        rate >= 0.5,
        "int8 lm_head argmax agreement collapsed to {rate:.3}; the kernel or quantization is broken"
    );
}

#[test]
#[ignore]
fn real_e4b_lever_ab_bench() {
    if std::env::var("NV_E4B_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_E4B_WGPU_TEST=1 to run");
        return;
    }
    let Some(_ctx) = ctx_or_skip() else { return };
    let dir = e4b_snapshot_dir();
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = e4b_host_weights_from_loader(&config, &loader).unwrap();

    let n_new: usize = std::env::var("NV_E4B_AB_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let reps: usize = std::env::var("NV_E4B_AB_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let bos = tok.token_to_id("<bos>").expect("<bos>");
    let mut prompt = vec![bos];
    prompt.extend(
        tok.encode(
            "Write a long, detailed essay about the history of navigation at sea, covering ancient, medieval, and modern techniques in depth.",
            false,
        )
        .unwrap()
        .get_ids()
        .to_vec(),
    );
    let max_seq = (prompt.len() + n_new).next_multiple_of(64) + 64;

    let configs: [(&str, &str, &str, &str); 6] = [
        ("off", "0", "0", "0"),
        ("sg16", "1", "0", "0"),
        ("int8", "0", "1", "0"),
        ("preenc", "0", "0", "1"),
        ("all", "1", "1", "1"),
        ("off-again", "0", "0", "0"),
    ];
    let mut lines = Vec::new();
    for rep in 0..reps {
        for (tag, sg, int8, pre) in configs.iter() {
            let guard = env_guard();
            std::env::set_var("NV_E4B_WGPU_W4_SG", sg);
            std::env::set_var("NV_E4B_LMHEAD_INT8", if *int8 == "1" { "1" } else { "0" });
            std::env::set_var("NV_E4B_WGPU_PREENC", if *pre == "1" { "1" } else { "0" });
            let mut m = Gemma4E4bWgpu::new(config.clone(), &host, max_seq).unwrap();
            std::env::remove_var("NV_E4B_WGPU_W4_SG");
            std::env::remove_var("NV_E4B_LMHEAD_INT8");
            std::env::remove_var("NV_E4B_WGPU_PREENC");
            drop(guard);
            let mut next = 0u32;
            for t in &prompt {
                next = m.decode_step(*t).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..n_new {
                next = m.decode_step(next).unwrap();
            }
            let ms = 1000.0 * t0.elapsed().as_secs_f64() / n_new as f64;
            let line = format!("[ab rep{rep}] {tag:<10} {ms:.2} ms/tok over {n_new} tok");
            eprintln!("{line}");
            lines.push(line);
        }
    }
    eprintln!("=== lever A/B summary ===");
    for l in &lines {
        eprintln!("{l}");
    }
}

fn top5(v: &[f32]) -> [u32; 5] {
    let mut idx = [0u32; 5];
    let mut val = [f32::NEG_INFINITY; 5];
    for (i, &x) in v.iter().enumerate() {
        if x > val[4] {
            let mut j = 4usize;
            while j > 0 && x > val[j - 1] {
                j -= 1;
            }
            let mut m = 4usize;
            while m > j {
                val[m] = val[m - 1];
                idx[m] = idx[m - 1];
                m -= 1;
            }
            val[j] = x;
            idx[j] = i as u32;
        }
    }
    idx
}

#[test]
#[ignore]
fn real_e4b_lmhead_int8_quality_eval() {
    if std::env::var("NV_E4B_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_E4B_WGPU_TEST=1 to run");
        return;
    }
    let Some(_ctx) = ctx_or_skip() else { return };
    let dir = e4b_snapshot_dir();
    eprintln!("E4B checkpoint: {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = e4b_host_weights_from_loader(&config, &loader).unwrap();

    let n_new: usize = std::env::var("NV_E4B_QE_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(320);
    let every: usize = std::env::var("NV_E4B_QE_LOGIT_EVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16)
        .max(1);
    let branch_len: usize = 32;
    let max_branches: usize = std::env::var("NV_E4B_QE_MAX_BRANCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let bos = tok.token_to_id("<bos>").expect("<bos>");
    let turn_close = tok.token_to_id("<turn|>").expect("<turn|>");
    let eos: std::collections::HashSet<u32> = [1, turn_close, 50].into_iter().collect();

    let completion_texts: [&str; 9] = [
        "Question: What is the capital of Australia?\nAnswer:",
        "The three primary colors of light are",
        "def is_prime(n):\n",
        "// C function that reverses a string in place\n#include <string.h>\n\nvoid reverse(char *s) {\n",
        "La Révolution française a commencé en",
        "Die Hauptstadt von Deutschland ist",
        "Érase una vez, en un pequeño pueblo junto al mar,",
        "The history of navigation at sea spans thousands of years, beginning with",
        "If a train travels at 60 miles per hour for 2.5 hours, the distance covered is",
    ];
    let mut prompts: Vec<Vec<u32>> = completion_texts
        .iter()
        .map(|s| {
            let mut ids = vec![bos];
            ids.extend(tok.encode(*s, false).unwrap().get_ids());
            ids
        })
        .collect();
    let chat_render =
        OfficialTemplate::load(&dir).render_user("Explain photosynthesis in simple terms.");
    prompts.push(
        tok.encode(chat_render.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec(),
    );
    let turn_open = tok.token_to_id("<|turn>").expect("<|turn>");
    assert!(
        prompts[9].contains(&turn_open) && prompts[9].contains(&turn_close),
        "gemma4 turn markers must resolve to single tokens, got {:?}",
        prompts[9]
    );
    assert!(
        prompts[9].first() == Some(&bos) && !prompts[9][1..].contains(&bos),
        "the chat prompt comes from the snapshot's own chat_template.jinja, which emits \
         {{{{ bos_token }}}} inline, so it must NOT also be prefixed the way the nine completion \
         prompts are -- exactly one <bos> at position 0. Rendered {chat_render:?} -> {:?}",
        prompts[9]
    );
    let max_prompt = prompts.iter().map(|p| p.len()).max().unwrap();
    let max_seq = (max_prompt + n_new + branch_len + 128).next_multiple_of(64);

    let build = |int8: bool| -> Gemma4E4bWgpu {
        let guard = env_guard();
        std::env::set_var("NV_E4B_LMHEAD_INT8", if int8 { "1" } else { "0" });
        let m = Gemma4E4bWgpu::new(config.clone(), &host, max_seq).unwrap();
        std::env::remove_var("NV_E4B_LMHEAD_INT8");
        drop(guard);
        m
    };

    let mut streams: Vec<Vec<u32>> = Vec::new();
    let mut bf_samples: Vec<Vec<(usize, Vec<f32>)>> = Vec::new();
    {
        let mut a = build(false);
        for (pi, prompt) in prompts.iter().enumerate() {
            a.reset();
            let mut next = a.prefill_prompt(prompt).unwrap();
            let mut stream = Vec::new();
            let mut samples = Vec::new();
            for step in 0..n_new {
                stream.push(next);
                if eos.contains(&next) || step + 1 == n_new {
                    break;
                }
                if step % every == 0 {
                    let (n2, lg) = a.decode_step_logits(next).unwrap();
                    samples.push((step, lg));
                    next = n2;
                } else {
                    next = a.decode_step(next).unwrap();
                }
            }
            eprintln!(
                "[bf16 p{pi}] {} tokens: {:?}",
                stream.len(),
                tok.decode(&stream, false).unwrap()
            );
            streams.push(stream);
            bf_samples.push(samples);
        }
    }

    let mut b = build(true);
    let mut total = 0usize;
    let mut agree = 0usize;
    let mut divergences: Vec<(usize, usize, u32, u32)> = Vec::new();
    let mut preds: Vec<Vec<u32>> = Vec::new();
    let mut max_ld_global = 0f32;
    let mut ld_sum = 0f64;
    let mut ov_sum = 0usize;
    let mut ov_min = 5usize;
    let mut n_samples = 0usize;
    for (pi, prompt) in prompts.iter().enumerate() {
        let stream = &streams[pi];
        b.reset();
        let mut pred = vec![b.prefill_prompt(prompt).unwrap()];
        for step in 0..stream.len().saturating_sub(1) {
            let feed = stream[step];
            if step % every == 0 {
                let (n2, lg) = b.decode_step_logits(feed).unwrap();
                let bf = &bf_samples[pi]
                    .iter()
                    .find(|(s, _)| *s == step)
                    .expect("aligned sample")
                    .1;
                let mut md = 0f32;
                for (x, y) in bf.iter().zip(lg.iter()) {
                    md = md.max((x - y).abs());
                }
                let ta = top5(bf);
                let tb = top5(&lg);
                let ov = ta.iter().filter(|i| tb.contains(i)).count();
                max_ld_global = max_ld_global.max(md);
                ld_sum += md as f64;
                ov_sum += ov;
                ov_min = ov_min.min(ov);
                n_samples += 1;
                pred.push(n2);
            } else {
                pred.push(b.decode_step(feed).unwrap());
            }
        }
        let mut p_agree = 0usize;
        for (k, (&bt, &it)) in stream.iter().zip(pred.iter()).enumerate() {
            if bt == it {
                p_agree += 1;
            } else {
                divergences.push((pi, k, bt, it));
            }
        }
        total += stream.len();
        agree += p_agree;
        eprintln!(
            "[agree p{pi}] {p_agree}/{} teacher-forced argmax agreement",
            stream.len()
        );
        preds.push(pred);
    }

    for (pi, prompt) in prompts.iter().enumerate() {
        b.reset();
        let mut next = b.prefill_prompt(prompt).unwrap();
        let mut free = Vec::new();
        for step in 0..n_new {
            free.push(next);
            if eos.contains(&next) || step + 1 == n_new {
                break;
            }
            next = b.decode_step(next).unwrap();
        }
        eprintln!(
            "[int8-free p{pi}] {} tokens: {:?}",
            free.len(),
            tok.decode(&free, false).unwrap()
        );
    }

    for (di, &(pi, k, bt, it)) in divergences.iter().enumerate() {
        if di >= max_branches {
            eprintln!(
                "[branch] skipping remaining {} divergences (cap {max_branches})",
                divergences.len() - di
            );
            break;
        }
        let stream = &streams[pi];
        let mut ctx_ids = prompts[pi].clone();
        ctx_ids.extend_from_slice(&stream[..k]);
        b.reset();
        let re = b.prefill_prompt(&ctx_ids).unwrap();
        if re != it {
            eprintln!("[branch p{pi}@{k}] replay pred {re} != recorded int8 pred {it}");
        }
        let mut branch = vec![it];
        let mut nx = b.decode_step(it).unwrap();
        for _ in 0..branch_len {
            branch.push(nx);
            if eos.contains(&nx) {
                break;
            }
            nx = b.decode_step(nx).unwrap();
        }
        let tail_from = k.saturating_sub(12);
        let bf_cont_end = (k + 1 + branch_len).min(stream.len());
        eprintln!(
            "[div {di} p{pi} step {k}] ctx tail {:?}\n  bf16 {bt} {:?} -> {:?}\n  int8 {it} {:?} -> {:?}",
            tok.decode(&stream[tail_from..k], false).unwrap(),
            tok.decode(&[bt], false).unwrap(),
            tok.decode(&stream[k..bf_cont_end], false).unwrap(),
            tok.decode(&[it], false).unwrap(),
            tok.decode(&branch, false).unwrap()
        );
    }

    let rate = agree as f64 / total as f64;
    eprintln!(
        "QE-SUMMARY agreement {agree}/{total} = {:.4}; divergences {}; logit samples {n_samples}: max|dlogit| {:.4} (mean of per-step max {:.4}), top5 overlap mean {:.3}/5 min {ov_min}/5",
        rate,
        divergences.len(),
        max_ld_global,
        ld_sum / n_samples.max(1) as f64,
        ov_sum as f64 / n_samples.max(1) as f64
    );
    assert!(
        total >= 2000,
        "eval too small: {total} tokens (want >= 2000)"
    );
    assert!(
        rate >= 0.98,
        "int8 lm_head teacher-forced agreement {rate:.4} below sanity floor"
    );
}

const E4B_CTX_TIMED_DECODE_STEPS_64_FOR_A_STABLE_MEAN: usize = 64;
const E4B_CTX_WARMUP_DECODE_STEPS_8_SO_FIRST_DISPATCHES_DONT_COUNT: usize = 8;

fn e4b_ctx_tokens_from_env_default_256_16k_128k() -> Vec<usize> {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => v
            .split(',')
            .map(|s| {
                let s = s.trim();
                let (num, mult) = match s.strip_suffix('k') {
                    Some(n) => (n, 1024usize),
                    None => (s, 1usize),
                };
                num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
            })
            .collect(),
        Err(_) => vec![256, 16 * 1024, 128 * 1024],
    }
}

#[test]
#[ignore]
fn real_e4b_ctx_prefill_tok_s_then_decode_ms_tok_vs_depth() {
    if std::env::var("NV_E4B_CTX_PREFILL_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_E4B_CTX_PREFILL_TEST=1 to run");
        return;
    }
    let Some(ctx) = ctx_or_skip() else { return };
    let dir = e4b_snapshot_dir();
    eprintln!("E4B checkpoint: {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let depths_requested = e4b_ctx_tokens_from_env_default_256_16k_128k();
    let depths: Vec<usize> = depths_requested
        .iter()
        .copied()
        .filter(|&d| {
            let fits = d <= config.max_position_embeddings;
            if !fits {
                eprintln!(
                    "NO-ROW depth={d}: exceeds max_position_embeddings={}",
                    config.max_position_embeddings
                );
            }
            fits
        })
        .collect();
    assert!(!depths.is_empty(), "every requested depth exceeded the model limit");
    let max_depth = depths.iter().copied().max().unwrap();

    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let t_load = std::time::Instant::now();
    let host = e4b_host_weights_from_loader(&config, &loader).unwrap();
    eprintln!("host weights loaded in {:.1}s", t_load.elapsed().as_secs_f64());
    drop(loader);
    let max_seq = max_depth
        + E4B_CTX_WARMUP_DECODE_STEPS_8_SO_FIRST_DISPATCHES_DONT_COUNT
        + E4B_CTX_TIMED_DECODE_STEPS_64_FOR_A_STABLE_MEAN
        + 16;
    let t_new = std::time::Instant::now();
    let mut m = Gemma4E4bWgpu::new(config.clone(), &host, max_seq).unwrap();
    drop(host);
    eprintln!(
        "Gemma4E4bWgpu::new: {:.1}s, {} decode passes, {} prefill passes, chunk {}",
        t_new.elapsed().as_secs_f64(),
        m.pass_count(),
        m.prefill_pass_count(),
        m.prefill_chunk_len()
    );
    assert!(
        m.prefill_chunk_len() >= 2,
        "chunked prefill is off; a step-primed number would not be a prefill number"
    );

    m.decode_step(5).unwrap();
    m.reset();
    m.prefill_chunk(&vec![5u32; m.prefill_chunk_len()]).unwrap();
    m.sync().unwrap();
    m.reset();

    for &depth in &depths {
        m.reset();
        let ids: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        let t0 = std::time::Instant::now();
        let mut token = m
            .prefill_prompt(&ids)
            .unwrap_or_else(|e| panic!("prefill_prompt at depth {depth}: {e:#}"));
        ctx.poll_blocking().expect("drain prefill work before stopping the clock");
        let prefill_s = t0.elapsed().as_secs_f64();
        assert_eq!(
            m.current_pos(),
            depth,
            "prefill must land exactly at the requested depth"
        );

        let mut step_ms: Vec<f64> = Vec::new();
        for i in 0..E4B_CTX_WARMUP_DECODE_STEPS_8_SO_FIRST_DISPATCHES_DONT_COUNT
            + E4B_CTX_TIMED_DECODE_STEPS_64_FOR_A_STABLE_MEAN
        {
            let t1 = std::time::Instant::now();
            token = m
                .decode_step(token)
                .unwrap_or_else(|e| panic!("timed step at depth {}: {e:#}", depth + i));
            if i >= E4B_CTX_WARMUP_DECODE_STEPS_8_SO_FIRST_DISPATCHES_DONT_COUNT {
                step_ms.push(t1.elapsed().as_secs_f64() * 1e3);
            }
        }
        let mean = step_ms.iter().sum::<f64>() / step_ms.len() as f64;
        step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = step_ms[step_ms.len() / 2];
        eprintln!(
            "CTX-PREFILL e4b-wgpu depth={depth} prefill_s={prefill_s:.3} prefill_tok_s={:.1} decode_mean_ms_tok={mean:.3} decode_median_ms_tok={median:.3} steps={}",
            depth as f64 / prefill_s,
            step_ms.len()
        );
    }
}
