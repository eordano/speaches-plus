#![cfg(feature = "wgpu")]

mod common;
use common::bf16_lin;
use common::have_gpu;
use common::LcgOddSeedShift33SignedUnit as Lcg;
use common::norm_vec;
use common::nozi_prof_dump;
use common::rel_err;
use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseConfig;
use nv_models::qwen3_5_moe::LayerType;
use nv_models::qwen3_5_moe_wgpu::HostBf16Lin;
use common::tiny_config_q3d_mixed_layers as tiny_config;
use common::tiny_weights_q3d as tiny_weights;

#[test]
fn tiny_wgpu_decode_matches_cpu_reference() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xd15e_9b00_0001);
    let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");
    eprintln!("[wgpu] recorded passes per token: {}", gpu.pass_count());

    let mut st = q3d::RefState::new(&cfg);
    let tokens: [u32; 6] = [3, 11, 5, 40, 2, 19];
    let mut worst_rel = 0f32;
    let mut top1_hits = 0usize;
    let mut all_logits: Vec<Vec<f32>> = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let (arg, logits) = gpu.decode_step_logits(*t).expect("decode step");
        all_logits.push(logits.clone());
        let want = q3d::reference_step(&cfg, &hw, &mut st, *t).expect("reference step");
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
        worst_rel = worst_rel.max(rel);
        eprintln!(
            "step {i}: tok={t} gpu_argmax={arg} ref_argmax={ref_arg} max_abs={abs:.6} rel={rel:.6}"
        );
        assert!(
            rel < 0.05,
            "step {i}: logits diverged from CPU reference (rel {rel})"
        );
    }
    eprintln!("[wgpu] worst relative logit error over 6 steps: {worst_rel:.6}");
    assert_eq!(
        top1_hits,
        tokens.len(),
        "argmax disagreed with the CPU reference on {} of {} steps",
        tokens.len() - top1_hits,
        tokens.len()
    );

    let (spread, _) = rel_err(&all_logits[0], &all_logits[2]);
    eprintln!("[wgpu] logit spread between step 0 and step 2: {spread:.6}");
    assert!(
        spread > 1e-3,
        "logits are insensitive to the input token / recurrent state (spread {spread}); \
         the reference comparison would then be vacuous"
    );
}

#[test]
fn tiny_wgpu_state_carries_and_resets() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xd15e_9b00_0002);
    let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");

    let (a0, l0) = gpu.decode_step_logits(7).expect("step");
    let (_a1, l1) = gpu.decode_step_logits(7).expect("step");
    let same = l0.iter().zip(l1.iter()).all(|(x, y)| (x - y).abs() <= 1e-6);
    assert!(
        !same,
        "feeding the same token twice produced identical logits: DeltaNet/KV state is not carried"
    );

    gpu.reset().expect("reset");
    let (a2, l2) = gpu.decode_step_logits(7).expect("step after reset");
    assert_eq!(a0, a2, "reset did not restore the first-token argmax");
    let (abs, _) = rel_err(&l0, &l2);
    assert!(
        abs <= 1e-5,
        "reset did not restore the first-token logits (max abs {abs})"
    );
    eprintln!("[wgpu] state carry verified; reset max_abs={abs:.8}");
}

#[test]
fn dense_config_parses_the_9b_shape_and_rejects_moe() {
    let raw = r#"{
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": false,
        "text_config": {
            "attn_output_gate": true,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "num_attention_heads": 16,
            "num_hidden_layers": 8,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "vocab_size": 248320,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25
            }
        }
    }"#;
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(raw).expect("parse dense config");
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.intermediate_size, 12288);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.rotary_dim(), 64);
    assert_eq!(cfg.rope_theta, 10_000_000.0);
    assert_eq!(cfg.eos_token_id, 248044);
    assert_eq!(cfg.bos_token_id, None);
    assert!(cfg.attn_output_gate);
    assert!(!cfg.tie_word_embeddings);
    assert_eq!(cfg.layer_types.len(), 8);
    assert_eq!(
        cfg.layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count(),
        2
    );

    let moeish = raw.replace(
        "\"intermediate_size\": 12288,",
        "\"intermediate_size\": 12288, \"num_experts\": 64,",
    );
    let err = Qwen3_5DenseConfig::from_hf_json_str(&moeish).unwrap_err();
    assert!(
        format!("{err}").contains("MoE"),
        "moe config was not rejected: {err}"
    );
}

#[test]
fn real_snapshot_config_is_supported_by_the_dense_wgpu_module() {
    let root = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots");
    let Ok(entries) = std::fs::read_dir(&root) else {
        eprintln!("[skip] {} not present", root.display());
        return;
    };
    let Some(snap) = entries
        .flatten()
        .map(|e| e.path().join("config.json"))
        .find(|p| p.exists())
    else {
        eprintln!(
            "[skip] no snapshot with config.json under {}",
            root.display()
        );
        return;
    };
    let cfg = Qwen3_5DenseConfig::from_hf_json_file(&snap).expect("parse config");
    eprintln!(
        "[cfg] hidden={} layers={} inter={} head_dim={} rot={} lin_k={}x{} lin_v={}x{}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.intermediate_size,
        cfg.head_dim,
        cfg.rotary_dim(),
        cfg.linear_num_key_heads,
        cfg.linear_key_head_dim,
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
    );
    assert_eq!(cfg.hidden_size % 4, 0);
    assert_eq!(cfg.intermediate_size % 2, 0);
    assert!(cfg.head_dim <= 256 && cfg.head_dim.is_multiple_of(2));
    assert!(cfg.linear_key_head_dim <= 128);
    assert!(cfg.linear_value_head_dim <= 128 && cfg.linear_value_head_dim.is_multiple_of(2));
    assert_eq!(cfg.rotary_dim() % 2, 0);
    assert_eq!(cfg.linear_num_value_heads % cfg.linear_num_key_heads, 0);
    assert_eq!(cfg.num_attention_heads % cfg.num_key_value_heads, 0);
    assert_eq!(cfg.layer_types.len(), cfg.num_hidden_layers);
    assert!(!cfg.tie_word_embeddings);
}

#[test]
#[ignore = "loads ~17 GB of bf16 weights; set NV_QWEN35_DENSE_WGPU_TEST=1"]
fn qwen35_dense_wgpu_real_weights_decode() {
    if std::env::var("NV_QWEN35_DENSE_WGPU_TEST").is_err() {
        eprintln!("[skip] NV_QWEN35_DENSE_WGPU_TEST not set");
        return;
    }
    if !have_gpu() {
        panic!("real-weights test needs a wgpu adapter");
    }
    let dir = match std::env::var("NV_QWEN35_DENSE_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            let root = std::path::PathBuf::from(std::env::var("HOME").unwrap())
                .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots");
            std::fs::read_dir(&root)
                .expect("snapshots dir")
                .flatten()
                .map(|e| e.path())
                .find(|p| p.join("config.json").exists())
                .expect("hydrated snapshot")
        }
    };
    let cfg = Qwen3_5DenseConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("loader");
    let t0 = std::time::Instant::now();
    let mut model =
        q3d::Qwen3_5DenseWgpu::from_loader(cfg.clone(), &loader, 512).expect("build model");
    eprintln!(
        "[real] loaded in {:.1}s, passes/token {}",
        t0.elapsed().as_secs_f64(),
        model.pass_count()
    );
    eprintln!("[real] {}", model.vram_report().render());

    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let prompt = "The capital of France is";
    let ids: Vec<u32> = tokenizer
        .encode(prompt, false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let mut last = 0u32;
    for t in &ids {
        last = model.decode_step(*t).expect("prefill step");
    }
    let n_new: usize = std::env::var("NV_QWEN35_DENSE_NEW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let mut out_ids: Vec<u32> = Vec::new();
    let t1 = std::time::Instant::now();
    for _ in 0..n_new {
        out_ids.push(last);
        if last == cfg.eos_token_id {
            break;
        }
        last = model.decode_step(last).expect("decode step");
    }
    let per_tok = t1.elapsed().as_secs_f64() * 1000.0 / out_ids.len().max(1) as f64;
    let text = tokenizer.decode(&out_ids, true).unwrap_or_default();
    eprintln!("[real] token_ids={out_ids:?}");
    eprintln!("[real] {per_tok:.1} ms/token; continuation: {text:?}");
    nozi_prof_dump();
    assert!(
        text.to_lowercase().contains("paris"),
        "greedy continuation of {prompt:?} did not mention Paris: {text:?}"
    );
}

#[test]
fn tiny_wgpu_chunked_prefill_matches_m1_loop_and_reference() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xd15e_9b00_0002);
    let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 48).expect("build wgpu model");
    if gpu.prefill_chunk_len() == 0 {
        eprintln!("[chunk] SKIP chunked prefill disabled (NV_WGPU_PREFILL_M)");
        return;
    }
    eprintln!(
        "[chunk] M={} prefill passes={}",
        gpu.prefill_chunk_len(),
        gpu.prefill_pass_count()
    );

    let tokens: Vec<u32> = (0..23u32).map(|i| (i * 7 + 3) % 64).collect();
    let (last, rest) = tokens.split_last().unwrap();

    for t in rest {
        gpu.prefill_step(*t).expect("m1 prefill step");
    }
    let (arg_l, logits_l) = gpu.decode_step_logits(*last).expect("m1 last step");

    gpu.reset().expect("reset");
    let done = gpu.prefill_tokens(rest).expect("prefill_tokens");
    eprintln!("[chunk] prefill_tokens consumed {done} of {}", rest.len());
    assert!(done > 0, "chunked prefill consumed nothing");
    for t in &rest[done..] {
        gpu.prefill_step(*t).expect("tail prefill step");
    }
    let (arg_c, logits_c) = gpu.decode_step_logits(*last).expect("chunked last step");

    assert_eq!(arg_l, arg_c, "argmax diverged chunked vs m=1 loop");
    let diff = logits_l
        .iter()
        .zip(logits_c.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!(
        "[chunk] logits bits: {diff} of {} lanes differ ({})",
        logits_l.len(),
        if diff == 0 { "BIT-IDENTICAL" } else { "drift" }
    );

    let mut st = q3d::RefState::new(&cfg);
    let mut want = Vec::new();
    for t in &tokens {
        want = q3d::reference_step(&cfg, &hw, &mut st, *t).expect("reference step");
    }
    let (abs, rel) = rel_err(&logits_c, &want);
    eprintln!("[chunk] vs CPU reference: max_abs={abs:.6} rel={rel:.6}");
    assert!(
        rel < 0.05,
        "chunked logits diverged from CPU reference (rel {rel})"
    );

    gpu.reset().expect("reset 2");
    let short: Vec<u32> = tokens[..5].to_vec();
    let (s_last, s_rest) = short.split_last().unwrap();
    let done = gpu.prefill_tokens(s_rest).expect("short prefill");
    for t in &s_rest[done..] {
        gpu.prefill_step(*t).expect("short tail");
    }
    let (arg_s, logits_s) = gpu.decode_step_logits(*s_last).expect("short last");
    gpu.reset().expect("reset 3");
    for t in s_rest {
        gpu.prefill_step(*t).expect("short m1");
    }
    let (arg_s1, logits_s1) = gpu.decode_step_logits(*s_last).expect("short m1 last");
    assert_eq!(
        arg_s, arg_s1,
        "short-prompt (tail-masked chunk) argmax diverged"
    );
    let sdiff = logits_s
        .iter()
        .zip(logits_s1.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!("[chunk] short-prompt logits bit diff lanes: {sdiff}");
}
