#![cfg(feature = "wgpu")]

mod common;
use common::have_gpu;
use common::LcgTop24TwoSided as Lcg;
use common::norm_tensor;
use common::rand_tensor_f32_shape as rand_tensor;
use common::real_snapshot;
use common::tiny_config_json;
use common::VOCAB_160 as VOCAB;
use candle_core::{Device, Tensor};
use nv_models::gemma4::LayerType;
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu;
use nv_weights::WeightLoader;
use std::collections::HashMap;
use common::TempDir;
use common::tensors_for_cfg;

fn temp_dir(tag: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("g4m_pf_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

fn run(
    gpu: &mut Gemma4MoeWgpu,
    prompt: &[u32],
    chunked: bool,
    n_new: usize,
) -> (Vec<u32>, Vec<f32>) {
    gpu.reset().unwrap();
    let last_idx = prompt.len() - 1;
    let done = if chunked {
        let d = gpu.prefill_tokens(&prompt[..last_idx]).unwrap();
        assert!(
            d > 0,
            "prefill_tokens consumed nothing -- chunked path never ran"
        );
        d
    } else {
        0
    };
    for t in &prompt[done..last_idx] {
        gpu.prefill_step(*t).unwrap();
    }
    let (mut tok, logits) = gpu.decode_step_logits(prompt[last_idx]).unwrap();
    let mut out = vec![tok];
    for _ in 0..n_new {
        tok = gpu.decode_step(tok).unwrap();
        out.push(tok);
    }
    (out, logits)
}

#[test]
fn chunked_prefill_matches_token_at_a_time() {
    if !have_gpu() {
        return;
    }
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let dir = temp_dir("match");
    let st = dir.0.join("model.safetensors");
    candle_core::safetensors::save(&tensors_for_cfg(&cfg, 0x9e37_79b9_7f4a_7c15), &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
    let mut gpu = Gemma4MoeWgpu::from_loader(cfg, &loader, 40).expect("build wgpu model");

    let m = gpu.prefill_chunk_len();
    assert!(
        m >= 2,
        "chunked prefill is disabled on this build (m={m}); the comparison would be vacuous"
    );
    eprintln!(
        "[pf] chunk m={m} prefill_passes={} decode_passes={}",
        gpu.prefill_pass_count(),
        gpu.pass_count()
    );

    let prompt: Vec<u32> = (0..19).map(|i| ((i * 7 + 3) % VOCAB) as u32).collect();
    assert!(
        prompt.len() - 1 > m,
        "prompt must span more than one chunk (len {} m {m})",
        prompt.len()
    );

    let (tok_a, logits_a) = run(&mut gpu, &prompt, false, 5);
    let (tok_b, logits_b) = run(&mut gpu, &prompt, true, 5);

    let spread = logits_a.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b))
        - logits_a.iter().fold(f32::INFINITY, |a, b| a.min(*b));
    assert!(
        spread > 1e-3,
        "logits are flat ({spread}); the comparison would not discriminate"
    );

    let worst = logits_a
        .iter()
        .zip(logits_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let exact = logits_a == logits_b;
    eprintln!("[pf] one-at-a-time {tok_a:?}");
    eprintln!("[pf] chunked       {tok_b:?}");
    eprintln!("[pf] worst |dlogit| = {worst:e} bit_exact={exact}");

    assert_eq!(
        tok_a, tok_b,
        "chunked prefill produced different tokens than the one-at-a-time path"
    );
    assert!(
        worst == 0.0,
        "chunked prefill logits are not bit-identical (worst {worst:e})"
    );
}

#[test]
#[ignore = "loads the real gemma-4-26B-A4B checkpoint; set NV_GEMMA4_MOE_WGPU_TEST=1"]
fn real_chunked_prefill_ab() {
    if std::env::var("NV_GEMMA4_MOE_WGPU_TEST").as_deref() != Ok("1") {
        eprintln!("[skip] NV_GEMMA4_MOE_WGPU_TEST not set");
        return;
    }
    if !have_gpu() {
        panic!("real-weights test needs a wgpu adapter");
    }
    let snap = real_snapshot();
    let mut cfg = Gemma4MoeConfig::from_hf_json_file(&snap.join("config.json")).unwrap();
    if let Some(n) = std::env::var("NV_GEMMA4_MOE_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        assert!(n > 0 && n <= cfg.base.num_hidden_layers);
        cfg.base.num_hidden_layers = n;
        cfg.base.layer_types.truncate(n);
        eprintln!("[real] TRUNCATED to the first {n} layers (partial load)");
    }
    let n_prompt: usize = std::env::var("NV_GEMMA4_MOE_PF_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(129);
    let max_seq: usize = std::env::var("NV_GEMMA4_MOE_MAX_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(n_prompt + 64);
    let vocab = cfg.base.vocab_size;

    let loader = nv_weights::WeightLoader::open_dir(&snap, &Device::Cpu).expect("open safetensors");
    let t0 = std::time::Instant::now();
    let mut gpu = Gemma4MoeWgpu::from_loader(cfg, &loader, max_seq).expect("build");
    eprintln!(
        "[real] built in {:.1}s, {} decode passes/token, {} prefill passes/chunk",
        t0.elapsed().as_secs_f64(),
        gpu.pass_count(),
        gpu.prefill_pass_count()
    );
    let m = gpu.prefill_chunk_len();
    assert!(
        m >= 2,
        "chunked prefill disabled (m={m}); A/B would be vacuous"
    );

    let prompt: Vec<u32> = (0..n_prompt)
        .map(|i| ((i * 1237 + 11) % (vocab - 8) + 4) as u32)
        .collect();
    let last_idx = prompt.len() - 1;

    gpu.reset().unwrap();
    let ta = std::time::Instant::now();
    for t in &prompt[..last_idx] {
        gpu.prefill_step(*t).unwrap();
    }
    let (tok_a, logits_a) = gpu.decode_step_logits(prompt[last_idx]).unwrap();
    let ms_a = ta.elapsed().as_secs_f64() * 1000.0;

    gpu.reset().unwrap();
    let tb = std::time::Instant::now();
    let done = gpu.prefill_tokens(&prompt[..last_idx]).unwrap();
    for t in &prompt[done..last_idx] {
        gpu.prefill_step(*t).unwrap();
    }
    let (tok_b, logits_b) = gpu.decode_step_logits(prompt[last_idx]).unwrap();
    let ms_b = tb.elapsed().as_secs_f64() * 1000.0;

    let worst = logits_a
        .iter()
        .zip(logits_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "[real] prompt {} tokens, chunk m={m}, chunked consumed {done}",
        prompt.len()
    );
    eprintln!(
        "[real] one-at-a-time prefill {ms_a:.1} ms ({:.3} ms/token)",
        ms_a / prompt.len() as f64
    );
    eprintln!(
        "[real] chunked       prefill {ms_b:.1} ms ({:.3} ms/token)  speedup {:.2}x",
        ms_b / prompt.len() as f64,
        ms_a / ms_b
    );
    eprintln!("[real] next token {tok_a} vs {tok_b}, worst |dlogit| = {worst:e}");
    assert_eq!(tok_a, tok_b, "chunked prefill changed the next token");
    assert!(
        worst == 0.0,
        "chunked prefill logits are not bit-identical (worst {worst:e})"
    );
}

#[test]
fn prefill_tokens_respects_kv_capacity() {
    if !have_gpu() {
        return;
    }
    let cfg = Gemma4MoeConfig::from_hf_json_str(&tiny_config_json()).unwrap();
    let dir = temp_dir("cap");
    let st = dir.0.join("model.safetensors");
    candle_core::safetensors::save(&tensors_for_cfg(&cfg, 0x51ee_d100_0007), &st).unwrap();
    let loader = WeightLoader::open_file(&st, &Device::Cpu).unwrap();
    let mut gpu = Gemma4MoeWgpu::from_loader(cfg, &loader, 20).expect("build wgpu model");
    let m = gpu.prefill_chunk_len();
    assert!(m >= 2, "chunked prefill disabled (m={m})");

    gpu.reset().unwrap();
    let toks: Vec<u32> = (0..40).map(|i| i % VOCAB as u32).collect();
    let done = gpu.prefill_tokens(&toks).unwrap();
    assert!(done <= 20, "prefilled {done} tokens past max_seq 20");
    assert_eq!(done, (20 / m) * m);
    assert_eq!(gpu.current_pos(), done);
    assert!(gpu.prefill_chunk(&toks[..m]).is_err() || gpu.current_pos() <= 20);
}
