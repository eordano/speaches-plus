#![cfg(feature = "wgpu")]

mod common;
use common::gow_tiny_weights as tiny_weights;
use common::have_gpu;
use common::LcgSplitMix64TwoSided as Lcg;
use common::mx_stack;
use common::nozi_prof_dump;
use common::rel_err;
use common::tiny_config_gpt_oss as tiny_config;
mod hub_snapshot;

use nv_models::gpt_oss_wgpu as gow;
use nv_models::gpt_oss_wgpu::{GptOssConfig, GptOssLayerType};

#[test]
fn mx_stack_expert_slices_roundtrip() {
    let mut r = Lcg::new(0x9055_0001);
    let st = mx_stack(&mut r, 3, 64, 64, 0.2);
    assert_eq!(st.e, 3);
    for e in 0..3 {
        let t = st.expert(e);
        assert_eq!(t.rows, 64);
        assert_eq!(t.cols, 64);
        let deq = t.dequantize();
        assert!(deq.iter().flatten().any(|v| *v != 0.0));
    }
    let t0 = st.expert(0).dequantize();
    let t1 = st.expert(1).dequantize();
    assert_ne!(t0, t1, "expert slices must not alias");
}

#[test]
fn yarn_tables_scale_cos_by_the_attention_factor() {
    let cfg = tiny_config();
    let (cos, sin) = gow::rope_tables(&cfg, 4);
    let mscale = cfg.attention_scaling();
    assert!(mscale > 1.0 && mscale < 1.5, "mscale {mscale}");
    let half = cfg.head_dim / 2;
    assert_eq!(cos.len(), 4 * half);
    for i in 0..half {
        assert!(
            (cos[i] - mscale).abs() < 1e-6,
            "pos 0 cos must equal mscale"
        );
        assert!(sin[i].abs() < 1e-6, "pos 0 sin must be zero");
    }
    let inv = gow::yarn_inv_freq(&cfg);
    let base: Vec<f32> = (0..half)
        .map(|i| 1.0 / cfg.rope_theta.powf((i as f32 * 2.0) / cfg.head_dim as f32))
        .collect();
    assert!(
        (inv[0] - base[0]).abs() / base[0] < 1e-5,
        "highest frequency must be extrapolated as-is"
    );
    let last = half - 1;
    assert!(
        inv[last] < base[last],
        "lowest frequency must be interpolated below the base rope"
    );
}

#[test]
fn tiny_wgpu_decode_matches_cpu_reference() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x6055_0001);
    let mut gpu = gow::GptOssWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");
    eprintln!("[wgpu] recorded passes per token: {}", gpu.pass_count());

    let mut st = gow::RefState::new(&cfg);
    let tokens: [u32; 7] = [3, 11, 5, 40, 2, 19, 33];
    let mut top1_hits = 0usize;
    let mut all_logits: Vec<Vec<f32>> = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let (arg, logits) = gpu.decode_step_logits(*t).expect("decode step");
        all_logits.push(logits.clone());
        let want = gow::reference_step(&cfg, &hw, &mut st, *t).expect("reference step");
        let (abs, rel) = rel_err(&logits, &want);
        let ref_arg = want
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        if arg == ref_arg {
            top1_hits += 1;
        }
        eprintln!(
            "step {i}: tok={t} gpu_argmax={arg} ref_argmax={ref_arg} max_abs={abs:.6} rel={rel:.6}"
        );
        assert!(
            rel < 0.05,
            "step {i}: logits diverged from CPU reference (rel {rel})"
        );
    }
    assert_eq!(
        top1_hits,
        tokens.len(),
        "argmax disagreed with the CPU reference on {} of {} steps",
        tokens.len() - top1_hits,
        tokens.len()
    );

    let (spread, _) = rel_err(&all_logits[0], &all_logits[2]);
    assert!(
        spread > 1e-3,
        "logits are insensitive to token/position (spread {spread}); the comparison is vacuous"
    );

    let steps_past_window = tokens.len() > cfg.sliding_window;
    assert!(
        steps_past_window,
        "test must decode past the sliding window ({} steps <= window {})",
        tokens.len(),
        cfg.sliding_window
    );
}

#[test]
fn tiny_wgpu_kv_state_carries_and_resets() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x6055_0002);
    let mut gpu = gow::GptOssWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");

    let (a0, l0) = gpu.decode_step_logits(7).expect("step");
    let (_a1, l1) = gpu.decode_step_logits(7).expect("step");
    let same = l0.iter().zip(l1.iter()).all(|(x, y)| (x - y).abs() <= 1e-6);
    assert!(
        !same,
        "feeding the same token twice produced identical logits: KV state is not carried"
    );

    gpu.reset().expect("reset");
    let (a2, l2) = gpu.decode_step_logits(7).expect("step after reset");
    assert_eq!(a0, a2, "reset did not restore the first-token argmax");
    let (abs, _) = rel_err(&l0, &l2);
    assert!(
        abs <= 1e-5,
        "reset did not restore the first-token logits (max abs {abs})"
    );
}

#[test]
fn sink_logits_shift_the_attention_denominator() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let mut hw = tiny_weights(&cfg, 0x6055_0003);
    let mut gpu = gow::GptOssWgpu::new(cfg.clone(), &hw, 32).expect("build");
    let (_t, base) = gpu.decode_step_logits(5).expect("step");

    for layer in hw.layers.iter_mut() {
        for s in layer.attn.sinks.iter_mut() {
            *s += 8.0;
        }
    }
    let mut gpu2 = gow::GptOssWgpu::new(cfg.clone(), &hw, 32).expect("build shifted");
    let (_t2, shifted) = gpu2.decode_step_logits(5).expect("step");
    let (abs, _) = rel_err(&base, &shifted);
    assert!(
        abs > 1e-4,
        "raising every sink by +8 must dampen attention output and move the logits (abs {abs})"
    );

    let mut st = gow::RefState::new(&cfg);
    let want = gow::reference_step(&cfg, &hw, &mut st, 5).expect("ref");
    let (_, rel) = rel_err(&shifted, &want);
    assert!(
        rel < 0.05,
        "shifted-sink logits diverged from reference (rel {rel})"
    );
}

fn gptoss_snapshot() -> Option<std::path::PathBuf> {
    hub_snapshot::dir_from_env_or_hub("NV_GPTOSS_DIR", "openai/gpt-oss-20b", &["config.json"])
}

fn gptoss_absent(test: &str) {
    hub_snapshot::precondition_absent(
        test,
        "no openai/gpt-oss-20b snapshot with config.json",
        "set NV_GPTOSS_DIR, or cache openai/gpt-oss-20b (it IS cached on this box)",
    );
}

#[test]
fn real_snapshot_config_is_supported_by_the_wgpu_module() {
    let Some(snap) = gptoss_snapshot() else {
        gptoss_absent("real_snapshot_config_is_supported_by_the_wgpu_module");
        return;
    };
    let cfg = GptOssConfig::from_hf_json_file(snap.join("config.json")).expect("parse config");
    assert_eq!(cfg.hidden_size, 2880);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.num_attention_heads, 64);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.head_dim, 64);
    assert_eq!(cfg.intermediate_size, 2880);
    assert_eq!(cfg.num_local_experts, 32);
    assert_eq!(cfg.num_experts_per_tok, 4);
    assert_eq!(cfg.vocab_size, 201088);
    assert_eq!(cfg.sliding_window, 128);
    assert_eq!(cfg.layer_types.len(), 24);
    assert_eq!(cfg.layer_types[0], GptOssLayerType::Sliding);
    assert_eq!(cfg.layer_types[1], GptOssLayerType::Full);
    assert!((cfg.yarn_factor - 32.0).abs() < 1e-6);
    assert_eq!(cfg.yarn_original_max, 4096);
    assert!(!cfg.tie_word_embeddings);
    assert_eq!(cfg.hidden_size % 64, 0);
    assert_eq!((2 * cfg.intermediate_size * cfg.hidden_size / 32) % 4, 0);
    let ms = cfg.attention_scaling();
    assert!(
        (ms - (0.1 * 32.0f32.ln() + 1.0)).abs() < 1e-6,
        "mscale {ms}"
    );
}

#[test]
fn real_snapshot_expert_tensors_have_the_expected_mxfp4_layout() {
    let Some(snap) = gptoss_snapshot() else {
        gptoss_absent("real_snapshot_expert_tensors_have_the_expected_mxfp4_layout");
        return;
    };
    if !snap.join("model.safetensors.index.json").exists() {
        hub_snapshot::precondition_absent(
            "real_snapshot_expert_tensors_have_the_expected_mxfp4_layout",
            &format!(
                "{}/model.safetensors.index.json not fetched",
                snap.display()
            ),
            "hydrate the gpt-oss-20b snapshot (the sharded index, not just config.json)",
        );
        return;
    }
    let raw = std::fs::read_to_string(snap.join("model.safetensors.index.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let wm = v.get("weight_map").and_then(|x| x.as_object()).unwrap();
    for name in [
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        "model.layers.0.mlp.experts.gate_up_proj_scales",
        "model.layers.0.mlp.experts.gate_up_proj_bias",
        "model.layers.0.mlp.experts.down_proj_blocks",
        "model.layers.0.mlp.experts.down_proj_scales",
        "model.layers.0.mlp.experts.down_proj_bias",
        "model.layers.0.self_attn.sinks",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.mlp.router.weight",
    ] {
        assert!(wm.contains_key(name), "index is missing {name}");
    }
}

#[test]
#[ignore = "loads ~13 GB of MXFP4 weights; set NV_GPTOSS_WGPU_TEST=1"]
fn gptoss_wgpu_real_weights_decode() {
    if std::env::var("NV_GPTOSS_WGPU_TEST").is_err() {
        eprintln!("[skip] NV_GPTOSS_WGPU_TEST not set");
        return;
    }
    if !have_gpu() {
        panic!("real-weights test needs a wgpu adapter");
    }
    let dir = gptoss_snapshot().expect("no gpt-oss-20b snapshot");
    let mut cfg = GptOssConfig::from_hf_json_file(dir.join("config.json")).expect("config");
    if let Some(n) = std::env::var("NV_GPTOSS_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        assert!(n > 0 && n <= cfg.num_hidden_layers);
        cfg.num_hidden_layers = n;
        cfg.layer_types.truncate(n);
        eprintln!("[real] TRUNCATED to the first {n} layers");
    }
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let max_seq: usize = std::env::var("NV_GPTOSS_MAX_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let t0 = std::time::Instant::now();
    let mut gpu =
        gow::GptOssWgpu::from_loader(cfg.clone(), &loader, max_seq).expect("build from loader");
    eprintln!(
        "[real] built in {:.1}s, {} passes/token",
        t0.elapsed().as_secs_f64(),
        gpu.pass_count()
    );
    eprint!("[real] {}", gpu.vram_report().render());

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let prompt = std::env::var("NV_GPTOSS_PROMPT")
        .unwrap_or_else(|_| "The capital of France is".to_string());
    let enc = tok.encode(prompt.as_str(), false).expect("encode");
    let ids: Vec<u32> = enc.get_ids().to_vec();
    assert!(!ids.is_empty());

    let mut next = gpu.prefill(&ids).expect("prefill");
    let mut out = vec![next];
    let n_new: usize = std::env::var("NV_GPTOSS_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let t1 = std::time::Instant::now();
    for _ in 1..n_new {
        next = gpu.decode_step(next).expect("decode");
        out.push(next);
    }
    let ms = t1.elapsed().as_secs_f64() * 1000.0 / (n_new.saturating_sub(1).max(1)) as f64;
    let text = tok.decode(&out, false).unwrap_or_default();
    eprintln!("[real] prompt={prompt:?}");
    eprintln!("[real] token_ids={out:?}");
    eprintln!("[real] continuation={text:?}");
    eprintln!("[real] {ms:.2} ms/tok decode");
    nozi_prof_dump();
    if std::env::var("NV_GPTOSS_LAYERS").is_ok() {
        eprintln!("[real] truncated model: coherence NOT expected, only that it runs");
        return;
    }
    assert!(
        out.iter().any(|t| *t != out[0]),
        "generation collapsed to a single repeated token"
    );
    if prompt == "The capital of France is" {
        assert!(
            text.contains("Paris"),
            "greedy continuation of the default prompt never names Paris: {text:?}"
        );
    }
}

const DECODE_TOKENS_256_THE_SCOREBOARD_CONTEXT: usize = 256;
const WARMUP_STEPS_32_REACH_THE_PIPELINE_PLATEAU: usize = 32;
const MAX_SEQ_512_HOLDS_PROMPT_PLUS_WARMUP_PLUS_256_TIMED_STEPS: usize = 512;

fn adapter_name() -> String {
    nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated probe must never silently skip")
        .info
        .name
        .clone()
}

fn model_at_rev(dir: &std::path::Path) -> String {
    let rev: String = dir
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.chars().take(7).collect())
        .unwrap_or_else(|| "local".into());
    format!("openai/gpt-oss-20b@{rev}")
}

#[test]
#[ignore = "measurement probe (run via nvk.sh probe): boots the ~13 GB MXFP4 checkpoint and \
            times 256 greedy decode steps; set NV_GPTOSS_WGPU_TEST=1"]
fn gptoss_wgpu_decode_tok_s_at_256_emits_the_canonical_measure_line() {
    if std::env::var("NV_GPTOSS_WGPU_TEST").is_err() {
        panic!(
            "this test is #[ignore]d, so it was asked for BY NAME, but NV_GPTOSS_WGPU_TEST=1 \
             is not set. This is a SKIP, not a pass."
        );
    }
    if !have_gpu() {
        panic!("measurement probe needs a wgpu adapter");
    }
    let dir = gptoss_snapshot().expect("no gpt-oss-20b snapshot");
    let cfg = GptOssConfig::from_hf_json_file(dir.join("config.json")).expect("config");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut gpu = gow::GptOssWgpu::from_loader(
        cfg,
        &loader,
        MAX_SEQ_512_HOLDS_PROMPT_PLUS_WARMUP_PLUS_256_TIMED_STEPS,
    )
    .expect("build from loader");

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let ids: Vec<u32> = tok
        .encode("The capital of France is", false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let mut next = gpu.prefill(&ids).expect("prefill");
    for _ in 0..WARMUP_STEPS_32_REACH_THE_PIPELINE_PLATEAU {
        next = gpu.decode_step(next).expect("warmup decode");
    }
    let mut step_ms = Vec::with_capacity(DECODE_TOKENS_256_THE_SCOREBOARD_CONTEXT);
    let mut out = Vec::with_capacity(DECODE_TOKENS_256_THE_SCOREBOARD_CONTEXT);
    for _ in 0..DECODE_TOKENS_256_THE_SCOREBOARD_CONTEXT {
        let t = std::time::Instant::now();
        next = gpu.decode_step(next).expect("decode");
        step_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        out.push(next);
    }
    let distinct: std::collections::BTreeSet<u32> = out.iter().copied().collect();
    assert!(
        distinct.len() >= 8,
        "timed decode is degenerate ({} distinct tokens over 256 steps); a number measured on \
         collapsed generation is not a number",
        distinct.len()
    );
    step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = step_ms[step_ms.len() / 2];
    nv_config::measure::Measurement {
        instrument: "gptoss-decode".into(),
        model_at_rev: model_at_rev(&dir),
        backend: "wgpu".into(),
        device: adapter_name(),
        batch: 1,
        tokens: DECODE_TOKENS_256_THE_SCOREBOARD_CONTEXT,
        steps: DECODE_TOKENS_256_THE_SCOREBOARD_CONTEXT,
        warmup: WARMUP_STEPS_32_REACH_THE_PIPELINE_PLATEAU,
        value: 1000.0 / median_ms,
        unit: "tok/s".into(),
        extras: Vec::new(),
    }
    .extra("basis", "greedy_decode_step_real_weights_prompt_primed")
    .extra("ms_tok_median", format!("{median_ms:.2}"))
    .emit();
}
